# sol_000188 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000163 (state 5ceb6a50) state=061cb89c sum of radii=2.627228 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_constraints(vars_arr, n):
    xs = vars_arr[:n]
    ys = vars_arr[n:2*n]
    rs = vars_arr[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    i, j = np.triu_indices(n, k=1)
    dx = xs[i] - xs[j]
    dy = ys[i] - ys[j]
    dr = rs[i] + rs[j]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def objective_func(vars_arr, n):
    """Objective: maximize sum of radii -> minimize negative sum"""
    return -np.sum(vars_arr[2*n:])

def solve_lp_radii(centers, n, i_idx, j_idx):
    """Solves LP to maximize sum of radii for fixed centers"""
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 0.0)
    bounds = [(0.0, lim) for lim in limits]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    m = len(i_idx)
    A_ub = np.zeros((m, n))
    b_ub = dists[i_idx, j_idx]
    
    for k, (u, v) in enumerate(zip(i_idx, j_idx)):
        A_ub[k, u] = 1.0
        A_ub[k, v] = 1.0
        
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def check_validity(centers, radii, n, i_idx, j_idx):
    """Strictly checks validity against optimization tolerance"""
    if np.any(centers[:, 0] < radii - 1e-9) or np.any(centers[:, 0] > 1.0 - radii + 1e-9):
        return False
    if np.any(centers[:, 1] < radii - 1e-9) or np.any(centers[:, 1] > 1.0 - radii + 1e-9):
        return False
    dx = centers[i_idx, 0] - centers[j_idx, 0]
    dy = centers[i_idx, 1] - centers[j_idx, 1]
    if np.any(dx**2 + dy**2 < (radii[i_idx] + radii[j_idx])**2 - 1e-9):
        return False
    return True

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': get_constraints, 'args': (n,)}
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Generate diverse initial configurations
    configs = []
    row_dists = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 5, 5, 6], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4],
        [5, 5, 6, 5, 5], [6, 5, 5, 6, 4], [5, 6, 4, 6, 5]
    ]
    
    for rd in row_dists:
        if sum(rd) < n: continue
        pts = []
        y = 0.085
        for idx, cnt in enumerate(rd):
            shift = 0.095 if idx % 2 == 1 else 0.0
            width = (cnt - 1) * 0.19
            x_start = 0.5 - width / 2.0 + shift
            for c in range(cnt):
                if len(pts) < n:
                    pts.append([x_start + c * 0.19, y])
            y += 0.1645
            if len(pts) >= n: break
        configs.append(np.array(pts[:n]))
        
    rng = np.random.default_rng(42)
    for _ in range(10):
        configs.append(rng.uniform(0.2, 0.8, (n, 2)))
        
    for cfg in configs:
        # Compute strictly feasible initial radius
        safe_r = np.min(np.minimum(np.minimum(cfg[:,0], 1.0-cfg[:,0]), np.minimum(cfg[:,1], 1.0-cfg[:,1])))
        diffs = cfg[:, np.newaxis, :] - cfg[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        safe_r = min(safe_r, np.min(dists)/2.0)
        safe_r = max(safe_r, 0.01)
        r_init = np.full(n, safe_r * 0.8)
        
        v0 = np.zeros(3 * n)
        v0[:n] = cfg[:, 0]
        v0[n:2*n] = cfg[:, 1]
        v0[2*n:] = r_init
        
        try:
            # Phase 1: Joint optimization
            res = minimize(objective_func, v0, args=(n,), method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13})
            
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2*n]
                centers = np.column_stack((cx, cy))
                
                # Phase 2: LP refinement for exact optimal radii
                r_lp, s_lp = solve_lp_radii(centers, n, i_idx, j_idx)
                if r_lp is not None and s_lp > best_sum:
                    if check_validity(centers, r_lp, n, i_idx, j_idx):
                        best_sum = s_lp
                        best_centers = centers.copy()
                        best_radii = r_lp.copy()
                        
                        # Phase 3: Iterative shake-and-optimize to escape local minima
                        v0_ref = np.zeros(3 * n)
                        v0_ref[:n] = centers[:, 0]
                        v0_ref[n:2*n] = centers[:, 1]
                        v0_ref[2*n:] = r_lp * 0.95
                        
                        for _ in range(6):
                            v0_ref[:2*n] += rng.uniform(-0.006, 0.006, 2*n)
                            v0_ref[:2*n] = np.clip(v0_ref[:2*n], 0.05, 0.95)
                            
                            res_ref = minimize(objective_func, v0_ref, args=(n,), method='SLSQP', 
                                               bounds=bounds, constraints=cons, 
                                               options={'maxiter': 5000, 'ftol': 1e-13})
                            if np.isfinite(res_ref.fun):
                                cx_r = res_ref.x[:n]
                                cy_r = res_ref.x[n:2*n]
                                centers_r = np.column_stack((cx_r, cy_r))
                                r_lp_r, s_lp_r = solve_lp_radii(centers_r, n, i_idx, j_idx)
                                if r_lp_r is not None and s_lp_r > best_sum:
                                    if check_validity(centers_r, r_lp_r, n, i_idx, j_idx):
                                        best_sum = s_lp_r
                                        best_centers = centers_r.copy()
                                        best_radii = r_lp_r.copy()
                                        v0_ref[2*n:] = r_lp_r * 0.95
        except Exception:
            continue
            
    # Fallback configuration
    if best_centers is None:
        best_centers = configs[0]
        r_fb, _ = solve_lp_radii(best_centers, n, i_idx, j_idx)
        best_radii = r_fb if r_fb is not None else np.full(n, 0.08)
        best_sum = np.sum(best_radii)
        
    # Final safety scaling to guarantee strict numerical validity
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
        scale = min(scale, np.min(d / np.maximum(rs, 1e-12)))
        
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
