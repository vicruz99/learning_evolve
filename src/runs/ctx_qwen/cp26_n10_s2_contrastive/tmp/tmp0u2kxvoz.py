import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

# Preallocate LP constraint matrix structure for speed
A_ub_lp = np.zeros((NUM_PAIRS, N))
A_ub_lp[np.arange(NUM_PAIRS), I_IDX] = 1.0
A_ub_lp[np.arange(NUM_PAIRS), J_IDX] = 1.0

def solve_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    dx = centers[I_IDX, 0] - centers[J_IDX, 0]
    dy = centers[I_IDX, 1] - centers[J_IDX, 1]
    b_ub = np.hypot(dx, dy)
    
    bounds = []
    for i in range(n):
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        bounds.append((0.0, max(1e-9, ub)))
        
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub_lp, b_ub=b_ub, bounds=bounds, method=method)
            if res.success:
                return np.maximum(res.x, 0.0)
        except Exception:
            continue
    return np.full(n, 0.001)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
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

def make_feasible(centers, radii):
    """Ensure strict feasibility within numerical tolerance."""
    for i in range(N):
        ub = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        if radii[i] > ub:
            radii[i] = max(0.0, ub - 1e-9)
            
    for _ in range(100):
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
    return centers, np.maximum(radii, 0.0)

def generate_init_configs(rng):
    """Generate diverse initial center configurations."""
    configs = []
    
    # Hexagonal lattices with varying spacing and vertical shifts
    for sp in np.linspace(0.18, 0.26, 8):
        for shift in [0.0, 0.05, 0.1]:
            c = np.zeros((N, 2))
            idx = 0
            y = 0.05 + shift
            row = 0
            while idx < N and y < 0.95:
                x = 0.05 + (row % 2) * sp / 2.0
                while x < 0.95 and idx < N:
                    c[idx] = [x, y]
                    idx += 1
                    x += sp
                y += sp * np.sqrt(3) / 2.0
                row += 1
            while idx < N:
                c[idx] = rng.uniform(0.1, 0.9, 2)
                idx += 1
            configs.append(c)
            
    # Square grids
    for step in np.linspace(0.17, 0.23, 6):
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
        configs.append(c)
        
    # Random uniform placements
    for _ in range(60):
        configs.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    inits = generate_init_configs(rng)
    
    # Phase 1: SLSQP from diverse starts to find strong local optima
    for c_init in inits:
        c_init = np.clip(c_init, 0.02, 0.98)
        r_init = solve_lp_radii(c_init) * 0.95
        r_init = np.maximum(r_init, 1e-4)
        x0 = np.zeros(3 * N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
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

    # Phase 2: Simulated Annealing on centers with LP evaluation
    # Escapes local minima by accepting temporary decreases in objective
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for step in range(3000):
            T = 0.003 * np.exp(-step / 1000.0)
            scale = 0.006 * np.exp(-step / 1200.0)
            
            c_pert = curr_c + rng.normal(0, scale, curr_c.shape)
            c_pert = np.clip(c_pert, 0.01, 0.99)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            
            accept = False
            if s_pert > curr_s:
                accept = True
            elif T > 1e-8:
                if rng.random() < np.exp((s_pert - curr_s) / T):
                    accept = True
                    
            if accept:
                curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
                    
                    # Polish new best with SLSQP
                    x0 = np.zeros(3 * N)
                    x0[0::3] = curr_c[:, 0]
                    x0[1::3] = curr_c[:, 1]
                    x0[2::3] = np.maximum(curr_r * 0.98, 1e-5)
                    try:
                        res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                       constraints=cons_opt, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                        if res.success:
                            c_pol = np.column_stack((res.x[0::3], res.x[1::3]))
                            r_pol = solve_lp_radii(c_pol)
                            s_pol = np.sum(r_pol)
                            if s_pol > curr_s:
                                curr_c, curr_r, curr_s = c_pol, r_pol, s_pol
                                best_sum = curr_s
                                best_centers = curr_c.copy()
                                best_radii = curr_r.copy()
                    except Exception:
                        pass

    # Phase 3: Local single-circle perturbations to relieve bottlenecks
    if best_centers is not None:
        for _ in range(1000):
            idx = rng.integers(N)
            c_pert = best_centers.copy()
            c_pert[idx] += rng.normal(0, 0.004, 2)
            c_pert = np.clip(c_pert, 0.02, 0.98)
            
            r_pert = solve_lp_radii(c_pert)
            s_pert = np.sum(r_pert)
            if s_pert > best_sum:
                best_sum = s_pert
                best_centers = c_pert.copy()
                best_radii = r_pert.copy()
                
                x0 = np.zeros(3*N)
                x0[0::3] = best_centers[:, 0]
                x0[1::3] = best_centers[:, 1]
                x0[2::3] = np.maximum(best_radii * 0.99, 1e-5)
                try:
                    res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                   constraints=cons_opt, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
                    c_pol = np.column_stack((res.x[0::3], res.x[1::3]))
                    r_pol = solve_lp_radii(c_pol)
                    if np.sum(r_pol) > best_sum:
                        best_sum = np.sum(r_pol)
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
    c_final, r_final = make_feasible(best_centers.copy(), best_radii.copy())
    return c_final, r_final, float(np.sum(r_final))