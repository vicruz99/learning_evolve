# sol_000254 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000227 (state bd5d11f3) state=0be3e82e sum of radii=2.619109 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers, n, i_idx, j_idx):
    """Solves LP to maximize sum of radii for fixed centers."""
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 0.0)
    bounds = [(0.0, lim) for lim in limits]
    
    dx = centers[i_idx, 0] - centers[j_idx, 0]
    dy = centers[i_idx, 1] - centers[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    m = len(i_idx)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), i_idx] = 1.0
    A_ub[np.arange(m), j_idx] = 1.0
    b_ub = dists
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return np.maximum(res.x, 1e-9), -res.fun
    except Exception:
        pass
    return np.full(n, 0.02), 0.0

def get_constraints(vars_arr, n, i_idx, j_idx):
    """Computes inequality constraints >= 0 for valid packing."""
    xs = vars_arr[:n]
    ys = vars_arr[n:2*n]
    rs = vars_arr[2*n:]
    
    # Boundary constraints
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dr = rs[i_idx] + rs[j_idx]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def objective_joint(vars_arr, n):
    """Objective: maximize sum of radii -> minimize negative sum"""
    return -np.sum(vars_arr[2*n:])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    i_idx, j_idx = np.triu_indices(n, k=1)
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': get_constraints, 'args': (n, i_idx, j_idx)}
    
    rng = np.random.default_rng(123)
    configs = []
    
    # Generate diverse hexagonal initial configurations
    row_dists = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4],
        [6, 5, 5, 6, 4], [5, 6, 4, 6, 5], [6, 6, 6, 4, 4],
        [5, 5, 5, 5, 5, 1], [7, 5, 6, 5, 3], [5, 5, 5, 6, 5],
        [6, 4, 6, 5, 5], [5, 6, 5, 5, 5], [4, 5, 6, 6, 5]
    ]
    
    for rd in row_dists:
        if sum(rd) < n: 
            continue
        pts = []
        y = 0.082
        for idx, cnt in enumerate(rd):
            shift = 0.088 if idx % 2 == 1 else 0.0
            width = (cnt - 1) * 0.182
            x_start = 0.5 - width / 2.0 + shift
            for c in range(cnt):
                if len(pts) < n:
                    pts.append([x_start + c * 0.182, y])
            y += 0.158
            if len(pts) >= n: 
                break
        configs.append(np.array(pts[:n]))
        
    # Add diverse random starts
    for _ in range(12):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Multi-start SLSQP + LP refinement
    for cfg in configs:
        r_init, _ = solve_lp_radii(cfg, n, i_idx, j_idx)
        r_init = np.maximum(r_init * 0.92, 0.015)
        
        v0 = np.zeros(3 * n)
        v0[:n] = cfg[:, 0]
        v0[n:2*n] = cfg[:, 1]
        v0[2*n:] = r_init
        
        try:
            res = minimize(objective_joint, v0, args=(n,), method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 12000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                centers = np.column_stack((res.x[:n], res.x[n:2*n]))
                r_lp, s_lp = solve_lp_radii(centers, n, i_idx, j_idx)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = centers.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Aggressive coordinate ascent local search on centers
    if best_centers is not None:
        for _ in range(4):  # Multiple passes to ensure convergence
            curr_centers = best_centers.copy()
            curr_radii = best_radii.copy()
            curr_sum = best_sum
            step = 0.028
            for _ in range(1200):
                idx = rng.integers(n)
                old_pos = curr_centers[idx].copy()
                curr_centers[idx] += rng.uniform(-step, step, 2)
                curr_centers[idx] = np.clip(curr_centers[idx], 0.001, 0.999)
                
                r_try, s_try = solve_lp_radii(curr_centers, n, i_idx, j_idx)
                if s_try > curr_sum + 1e-9:
                    curr_sum = s_try
                    curr_radii = r_try.copy()
                    if s_try > best_sum:
                        best_sum = s_try
                        best_centers = curr_centers.copy()
                        best_radii = curr_radii.copy()
                else:
                    curr_centers[idx] = old_pos
                step *= 0.9975
                
    # Phase 3: Final SLSQP polish from the best local search result
    if best_centers is not None:
        v0_polish = np.zeros(3 * n)
        v0_polish[:n] = best_centers[:, 0]
        v0_polish[n:2*n] = best_centers[:, 1]
        v0_polish[2*n:] = best_radii * 0.96
        
        try:
            res_pol = minimize(objective_joint, v0_polish, args=(n,), method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14})
            if np.isfinite(res_pol.fun):
                c_pol = np.column_stack((res_pol.x[:n], res_pol.x[n:2*n]))
                r_pol, s_pol = solve_lp_radii(c_pol, n, i_idx, j_idx)
                if s_pol > best_sum:
                    best_sum = s_pol
                    best_centers = c_pol.copy()
                    best_radii = r_pol.copy()
        except Exception:
            pass
            
    # Phase 4: Strict safety scaling to guarantee validation tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    dx = best_centers[i_idx, 0] - best_centers[j_idx, 0]
    dy = best_centers[i_idx, 1] - best_centers[j_idx, 1]
    d = np.sqrt(dx**2 + dy**2)
    rs = best_radii[i_idx] + best_radii[j_idx]
    if np.any(rs > 1e-12):
        valid_mask = rs > 1e-12
        scale = min(scale, np.min(d[valid_mask] / rs[valid_mask]))
        
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
