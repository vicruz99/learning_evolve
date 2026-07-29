import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
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
    
    bounds = []
    for i in range(n):
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(1e-9, ub)))
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return np.full(n, 0.005)

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

def generate_init_configs(rng):
    """Generate diverse initial configurations."""
    inits = []
    
    # 1. Hexagonal lattices with varying spacings and offsets
    for seed in range(25):
        r_gen = np.random.RandomState(seed)
        c = np.zeros((N, 2))
        idx = 0
        sp = 0.14 + r_gen.uniform(0.0, 0.06)
        y = sp / 2 + r_gen.uniform(0, 0.02)
        row = 0
        while idx < N and y < 1.0 - sp / 2:
            x = sp / 2 + (row % 2) * sp / 2
            while x < 1.0 - sp / 2 and idx < N:
                c[idx] = [x, y]
                idx += 1
                x += sp
            y += sp * np.sqrt(3) / 2.0
            row += 1
        while idx < N:
            c[idx] = r_gen.uniform(0.1, 0.9, 2)
            idx += 1
        c += r_gen.normal(0, 0.005, c.shape)
        inits.append(np.clip(c, 0.01, 0.99))
        
    # 2. Staggered row patterns (known to pack well)
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [7,6,5,4,4], [4,5,6,5,6], [8,6,5,4,3], [6,6,5,5,4], [7,7,5,4,3], [5,5,5,5,6]]
    for pat in patterns:
        pts = []
        y = 0.04
        dy = 0.92 / (len(pat) - 0.5)
        for r_idx, cnt in enumerate(pat):
            shift = 0.0 if r_idx % 2 == 0 else 0.07
            x = 0.04 + shift
            step = 0.92 / (cnt - 0.5) if cnt > 1 else 0.0
            for _ in range(cnt):
                if len(pts) < N:
                    pts.append([x, y])
                x += step
            y += dy
        while len(pts) < N:
            pts.append([0.5, 0.5])
        arr = np.array(pts[:N]) + rng.normal(0, 0.004, (N, 2))
        inits.append(np.clip(arr, 0.02, 0.98))
        
    # 3. Random uniform placements
    for _ in range(40):
        inits.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = generate_init_configs(rng)
    
    # Phase 1: Multi-start SLSQP optimization
    for c0 in inits:
        r0 = solve_lp_radii(c0) * 0.96
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = np.maximum(r0, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if res.success or -res.fun > best_sum:
                c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
                r_opt = solve_lp_radii(c_opt)
                s_opt = np.sum(r_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            pass

    # Phase 2: Adaptive Basin Hopping with LP evaluation
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(3500):
            # Decaying noise schedule
            scale = 0.006 * np.exp(-step / 1200.0)
            c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.005, 0.995)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            
            # Accept if improves current or best
            if s_pert > curr_s:
                curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
                
                # Polish new best with SLSQP
                x0_p = np.zeros(3 * N)
                x0_p[0::3] = curr_c[:, 0]
                x0_p[1::3] = curr_c[:, 1]
                x0_p[2::3] = np.maximum(curr_r * 0.99, 1e-5)
                
                try:
                    res_p = minimize(objective, x0_p, method='SLSQP', bounds=bounds_opt,
                                     constraints=cons_opt, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                    if res_p.success:
                        c_pol = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                        r_pol = solve_lp_radii(c_pol)
                        s_pol = np.sum(r_pol)
                        if s_pol > best_sum:
                            best_sum = s_pol
                            best_centers = c_pol.copy()
                            best_radii = r_pol.copy()
                            curr_c, curr_r, curr_s = best_centers, best_radii, best_sum
                except Exception:
                    pass
                    
    # Phase 3: Coordinate-wise Perturbation (moves 1-2 circles at a time)
    if best_centers is not None:
        for _ in range(2500):
            n_pert = rng.choice([1, 2])
            idxs = rng.choice(N, n_pert, replace=False)
            c_pert = best_centers.copy()
            c_pert[idxs] += rng.normal(0, 0.0025, (n_pert, 2))
            c_pert = np.clip(c_pert, 0.005, 0.995)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = c_pert.copy()
                best_radii = r_pert.copy()
                
                # Quick polish
                x0 = np.zeros(3 * N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.99, 1e-5)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt, options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                    if res.success:
                        c_pol = np.column_stack((res.x[0::3], res.x[1::3]))
                        r_pol = solve_lp_radii(c_pol)
                        s_pol = np.sum(r_pol)
                        if s_pol > best_sum:
                            best_sum = s_pol
                            best_centers = c_pol.copy()
                            best_radii = r_pol.copy()
                except Exception:
                    pass

    # Fallback safety net
    if best_centers is None:
        best_centers = inits[0]
        best_radii = solve_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 4: Strict post-processing to guarantee validator compliance
    c_final, r_final = make_strictly_feasible(best_centers.copy(), best_radii.copy())
    
    return c_final, r_final, float(np.sum(r_final))