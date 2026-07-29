import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    ub = np.minimum.reduce([centers[:, 0], 1.0 - centers[:, 0], centers[:, 1], 1.0 - centers[:, 1]])
    bounds = [(0.0, max(0.0, u)) for u in ub]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def objective_joint(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints_joint(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap (>= 0)."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4 * N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2 * N] = 1.0 - cx - r
    c[2 * N:3 * N] = cy - r
    c[3 * N:4 * N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4 * N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def make_strictly_feasible(centers, radii):
    """Deterministically resolve overlaps and boundary violations."""
    for i in range(N):
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > ub:
            radii[i] = max(0.0, ub - 1e-9)
            
    for _ in range(150):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if d < radii[i] + radii[j] - 1e-12:
                exc = radii[i] + radii[j] - d
                radii[i] -= exc * 0.5
                radii[j] -= exc * 0.5
                changed = True
        if not changed:
            break
    radii = np.maximum(radii, 0.0)
    return centers, radii

def generate_initializations(rng):
    """Generate diverse initial center configurations."""
    inits = []
    # 1. Hexagonal lattices with varying spacings and offsets
    for sp in np.linspace(0.16, 0.28, 10):
        for shift_y in [0.0, 0.03, 0.07]:
            c = np.zeros((N, 2))
            idx = 0
            y = 0.04 + shift_y
            row = 0
            while idx < N and y < 0.96:
                x = 0.04 + (row % 2) * sp / 2.0
                while x < 0.96 and idx < N:
                    c[idx] = [x, y]
                    idx += 1
                    x += sp
                y += sp * np.sqrt(3) / 2.0
                row += 1
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
            inits.append(c + rng.normal(0, 0.005, c.shape))
            
    # 2. Square grids
    for step in np.linspace(0.15, 0.25, 8):
        c = np.zeros((N, 2))
        idx = 0
        y = 0.05
        while y < 0.95 and idx < N:
            x = 0.05
            while x < 0.95 and idx < N:
                c[idx] = [x, y]
                idx += 1
                x += step
            y += step
        while idx < N:
            c[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        inits.append(c + rng.normal(0, 0.004, c.shape))
        
    # 3. Random uniform placements
    for _ in range(30):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints_joint}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = generate_initializations(rng)
    
    # Phase 1: Multi-start SLSQP with LP initialization
    for base in inits:
        c_init = np.clip(base, 0.02, 0.98)
        r_init, _ = solve_lp_radii(c_init)
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = np.maximum(r_init * 0.98, 1e-5)
        
        try:
            res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt,
                           options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if res.success or -res.fun > best_sum:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt, s_opt = solve_lp_radii(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            pass

    # Phase 2: Intensive Hill-Climbing on Centers using LP evaluation
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        # Adaptive cooling schedule
        for step in range(3500):
            scale = 0.008 * np.exp(-step / 1200.0) + 0.0002
            c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.005, 0.995)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            
            # Accept if better, or probabilistically if slightly worse (SA)
            if s_pert > curr_s or rng.random() < np.exp((s_pert - curr_s) / 0.01):
                curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                
                if s_pert > best_sum:
                    best_sum = s_pert
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
                    
                    # Periodic SLSQP polish on new best
                    if step % 200 == 0:
                        x0_p = np.zeros(3 * N)
                        x0_p[0::3] = curr_c[:, 0]
                        x0_p[1::3] = curr_c[:, 1]
                        x0_p[2::3] = np.maximum(curr_r * 0.99, 1e-5)
                        try:
                            res_p = minimize(objective_joint, x0_p, method='SLSQP', bounds=bounds_opt,
                                             constraints=cons_opt,
                                             options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                            if res_p.success:
                                c_pol = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                                r_pol, s_pol = solve_lp_radii(c_pol)
                                if s_pol > curr_s:
                                    curr_c, curr_r, curr_s = c_pol, r_pol, s_pol
                                    best_sum = curr_s
                                    best_centers = curr_c.copy()
                                    best_radii = curr_r.copy()
                        except Exception:
                            pass

    # Phase 3: Targeted Subset Perturbations to escape shallow minima
    if best_centers is not None:
        for _ in range(800):
            num_pert = rng.choice([2, 3, 4, 5])
            idxs = rng.choice(N, num_pert, replace=False)
            c_pert = best_centers.copy()
            noise = rng.uniform(0.001, 0.006)
            c_pert[idxs] += rng.normal(0, noise, (num_pert, 2))
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert, s_pert = solve_lp_radii(c_pert)
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = c_pert.copy()
                best_radii = r_pert.copy()
                
                # Quick polish
                x0 = np.zeros(3*N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.99, 1e-5)
                try:
                    res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt,
                                   options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
                    c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_opt, s_opt = solve_lp_radii(c_opt)
                    if s_opt > best_sum:
                        best_sum = s_opt
                        best_centers = c_opt.copy()
                        best_radii = r_opt.copy()
                except Exception:
                    pass

    # Phase 4: Boundary Push Heuristic
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        for _ in range(200):
            i = rng.integers(0, N)
            # Push towards nearest boundary/corner
            dirs = np.sign(np.array([0.5 - curr_c[i, 0], 0.5 - curr_c[i, 1]]) * rng.uniform(-1, 1, 2))
            step = rng.uniform(0.002, 0.01)
            c_push = curr_c.copy()
            c_push[i] += dirs * step
            c_push = np.clip(c_push, 0.005, 0.995)
            
            r_push, s_push = solve_lp_radii(c_push)
            if s_push > curr_s:
                curr_c, curr_r, curr_s = c_push, r_push, s_push
                if s_push > best_sum:
                    best_sum = s_push
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()

    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Phase 5: Strict post-processing to guarantee validator compliance
    c_final, r_final = make_strictly_feasible(best_centers.copy(), best_radii.copy())
    
    # Final strict boundary clamp
    for i in range(N):
        ub = min(c_final[i, 0], 1.0 - c_final[i, 0], c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], ub - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    return c_final, r_final, float(np.sum(r_final))