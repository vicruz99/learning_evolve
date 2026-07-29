# sol_000120 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000091 (state 4dfa0868) state=4a72b437 sum of radii=2.608631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)
NUM_PAIRS = len(I_IDX)

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    # Compute pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.hypot(diff[:, :, 0], diff[:, :, 1])
    
    A_ub = np.zeros((NUM_PAIRS, n))
    A_ub[np.arange(NUM_PAIRS), I_IDX] = 1.0
    A_ub[np.arange(NUM_PAIRS), J_IDX] = 1.0
    b_ub = dists[I_IDX, J_IDX]
    
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, mx)))
        
    for method in ['highs', 'interior-point']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and np.all(res.x >= -1e-9):
                return np.maximum(res.x, 0.0)
        except Exception:
            continue
    return np.full(n, 0.01)

def objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """Inequality constraints: boundary clearance and pairwise non-overlap."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = np.empty(4*N + NUM_PAIRS)
    c[:N] = cx - r
    c[N:2*N] = 1.0 - cx - r
    c[2*N:3*N] = cy - r
    c[3*N:4*N] = 1.0 - cy - r
    
    dx = cx[I_IDX] - cx[J_IDX]
    dy = cy[I_IDX] - cy[J_IDX]
    c[4*N:] = np.hypot(dx, dy) - (r[I_IDX] + r[J_IDX])
    return c

def generate_initial_configs(rng):
    """Generate diverse structured initial configurations."""
    configs = []
    
    # 1. Hexagonal lattices with varying spacing
    for s in np.linspace(0.14, 0.22, 8):
        centers = np.zeros((N, 2))
        idx = 0
        row = 0
        y = s / 2
        while idx < N and y < 1.0 - s / 2:
            x_start = s / 2 + (row % 2) * s / 2
            col = 0
            while x_start + col * s < 1.0 - s / 2 and idx < N:
                centers[idx] = [x_start + col * s, y]
                idx += 1
                col += 1
            y += s * np.sqrt(3) / 2
            row += 1
        while idx < N:
            centers[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        configs.append(centers)
        
    # 2. Square grids
    for s in np.linspace(0.15, 0.22, 6):
        centers = np.zeros((N, 2))
        idx = 0
        y = s / 2
        while y < 1.0 - s / 2 and idx < N:
            x = s / 2
            while x < 1.0 - s / 2 and idx < N:
                centers[idx] = [x, y]
                x += s
                idx += 1
            y += s
        while idx < N:
            centers[idx] = rng.uniform(0.1, 0.9, 2)
            idx += 1
        configs.append(centers)
        
    # 3. Corner/Edge biased patterns
    for _ in range(10):
        centers = np.zeros((N, 2))
        centers[:4] = rng.uniform(0.02, 0.15, (4, 2))
        centers[4:8] = rng.uniform(0.85, 0.98, (4, 2))
        centers[8:] = rng.uniform(0.1, 0.9, (N-8, 2))
        configs.append(centers)
        
    # 4. Pure random
    for _ in range(15):
        configs.append(rng.uniform(0.05, 0.95, (N, 2)))
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_opt = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng_main = np.random.default_rng(42)
    inits = generate_initial_configs(rng_main)
    
    # Phase 1: Multi-start SLSQP to find strong basins
    for base in inits:
        c_init = base + rng_main.normal(0, 0.004, base.shape)
        c_init = np.clip(c_init, 0.02, 0.98)
        
        r_init = solve_radii_lp(c_init) * 0.95
        x0 = np.zeros(3*N)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = np.maximum(r_init, 1e-5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                           constraints=cons_opt, 
                           options={'maxiter': 15000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
                curr_r = solve_radii_lp(curr_c)
                curr_s = np.sum(curr_r)
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = curr_r.copy()
        except Exception:
            pass

    # Phase 2: Stochastic Hill-Climbing on centers using exact LP evaluation
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        # Adaptive perturbation scales for coarse-to-fine search
        scales = [0.006, 0.0025, 0.0008, 0.0002]
        for scale in scales:
            for _ in range(400):
                c_pert = curr_c + rng_main.normal(0, scale, curr_c.shape)
                c_pert = np.clip(c_pert, 0.01, 0.99)
                
                r_pert = solve_radii_lp(c_pert)
                s_pert = np.sum(r_pert)
                
                if s_pert > curr_s:
                    curr_c, curr_r, curr_s = c_pert, r_pert, s_pert
                    if curr_s > best_sum:
                        best_sum = curr_s
                        best_centers = curr_c.copy()
                        best_radii = curr_r.copy()
                        
                        # Local SLSQP polishing after improvement
                        x0 = np.zeros(3*N)
                        x0[0::3] = curr_c[:, 0]
                        x0[1::3] = curr_c[:, 1]
                        x0[2::3] = np.maximum(curr_r * 0.98, 1e-5)
                        try:
                            res = minimize(objective, x0, method='SLSQP', bounds=bounds_opt,
                                           constraints=cons_opt,
                                           options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                            if res.success:
                                c_ref = np.column_stack((res.x[0::3], res.x[1::3]))
                                r_ref = solve_radii_lp(c_ref)
                                s_ref = np.sum(r_ref)
                                if s_ref > curr_s:
                                    curr_c, curr_r, curr_s = c_ref, r_ref, s_ref
                                    best_sum = curr_s
                                    best_centers = curr_c.copy()
                                    best_radii = curr_r.copy()
                        except Exception:
                            pass

    # Fallback safety net
    if best_centers is None:
        best_centers = rng_main.uniform(0.1, 0.9, (N, 2))
        best_radii = solve_radii_lp(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 3: Strict post-processing to guarantee numerical validity
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Enforce boundary constraints strictly
    for i in range(N):
        mx = min(c_final[i, 0], 1.0 - c_final[i, 0], 
                 c_final[i, 1], 1.0 - c_final[i, 1])
        r_final[i] = min(r_final[i], mx - 1e-9)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any remaining numerical overlaps
    for _ in range(50):
        changed = False
        for k in range(NUM_PAIRS):
            i, j = I_IDX[k], J_IDX[k]
            d = np.hypot(c_final[i,0]-c_final[j,0], c_final[i,1]-c_final[j,1])
            if d < r_final[i] + r_final[j] - 1e-11:
                exc = r_final[i] + r_final[j] - d
                r_final[i] -= exc * 0.5
                r_final[j] -= exc * 0.5
                r_final[i] = max(0.0, r_final[i])
                r_final[j] = max(0.0, r_final[j])
                changed = True
        if not changed:
            break
            
    return c_final, r_final, float(np.sum(r_final))
