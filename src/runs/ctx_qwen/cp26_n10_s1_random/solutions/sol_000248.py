# sol_000248 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000238 (state ed5af233) state=cb25a8f0 sum of radii=1.523965 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers, n, idx_i, idx_j, A_lp_pairs):
    """Solves LP to maximize sum of radii for fixed centers."""
    wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    wall = np.maximum(wall, 1e-9)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.concatenate([wall, dists[idx_i, idx_j]])
    A_ub = np.vstack([np.eye(n), A_lp_pairs])
    bounds = [(0.0, w) for w in wall]
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def cons_equal(vars_arr, n):
    """Constraints for equal-radius packing optimization."""
    c = vars_arr[:2*n].reshape(n, 2)
    t = vars_arr[-1]
    
    cx, cy = c[:, 0], c[:, 1]
    c_list = [cx - t, 1.0 - cx - t, cy - t, 1.0 - cy - t]
    
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    idx = np.triu_indices(n, k=1)
    c_list.append(dist_sq[idx] - 4.0 * t**2)
    return np.concatenate(c_list)

def cons_joint(vars_arr, n):
    """Constraints for joint center-radius optimization."""
    cx = vars_arr[:n]
    cy = vars_arr[n:2*n]
    r = vars_arr[2*n:]
    
    c_list = [cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r]
    
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    dr = r[:, np.newaxis] + r[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    idx = np.triu_indices(n, k=1)
    c_list.append(dist_sq[idx] - dr[idx]**2)
    return np.concatenate(c_list)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(2025)
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    A_lp_pairs = np.zeros((len(idx_i), n))
    A_lp_pairs[np.arange(len(idx_i)), idx_i] = 1.0
    A_lp_pairs[np.arange(len(idx_i)), idx_j] = 1.0
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # 1. Generate diverse initial configurations
    configs = []
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [6,6,4,6,4], [4,6,6,5,5], [5,5,6,5,5]]
    for pat in patterns:
        pts = []
        y = 0.1
        r0 = 0.1
        for i, cnt in enumerate(pat):
            shift = r0 if i % 2 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        pts = np.array(pts[:n])
        mn = pts.min(axis=0); mx = pts.max(axis=0)
        pts = 0.1 + 0.8 * (pts - mn) / (mx - mn + 1e-9)
        configs.append(pts)
        
    for _ in range(8):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # 2. Force-directed layout to find tight packings
    for cfg in configs:
        c = cfg.copy()
        r = np.full(n, 0.085)
        
        for step in range(1500):
            r *= 1.00008
            forces = np.zeros_like(c)
            
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
            np.fill_diagonal(dists, np.inf)
            overlap = np.maximum(0.0, r[:, np.newaxis] + r[np.newaxis, :] - dists)
            rep = overlap / dists
            forces[:, 0] += np.sum(diff[:, :, 0] * rep, axis=1) * 4.0
            forces[:, 1] += np.sum(diff[:, :, 1] * rep, axis=1) * 4.0
            
            for i in range(n):
                if c[i,0] - r[i] < 0: forces[i,0] += (r[i] - c[i,0]) * 10.0
                if c[i,0] + r[i] > 1.0: forces[i,0] -= (c[i,0] + r[i] - 1.0) * 10.0
                if c[i,1] - r[i] < 0: forces[i,1] += (r[i] - c[i,1]) * 10.0
                if c[i,1] + r[i] > 1.0: forces[i,1] -= (c[i,1] + r[i] - 1.0) * 10.0
                
            c += forces * 0.015
            c = np.clip(c, 0.0, 1.0)
            
        lp_r, lp_s = solve_lp_radii(c, n, idx_i, idx_j, A_lp_pairs)
        if lp_r is not None and lp_s > best_sum:
            best_sum = lp_s
            best_centers = c.copy()
            best_radii = lp_r.copy()
            
    # 3. Phase A: Optimize equal radius centers to find well-spread layouts
    for cfg in configs:
        x0_eq = np.concatenate([cfg.flatten(), [0.09]])
        bounds_eq = [(0.0, 1.0)] * (2*n) + [(0.05, 0.13)]
        try:
            res_eq = minimize(lambda v: -v[-1], x0_eq, method='SLSQP', bounds=bounds_eq,
                              constraints={'type': 'ineq', 'fun': cons_equal, 'args': (n,)},
                              options={'maxiter': 5000, 'ftol': 1e-13})
            if np.isfinite(res_eq.fun):
                c_opt = res_eq.x[:2*n].reshape(n, 2)
                lp_r, lp_s = solve_lp_radii(c_opt, n, idx_i, idx_j, A_lp_pairs)
                if lp_r is not None and lp_s > best_sum:
                    best_sum = lp_s
                    best_centers = c_opt.copy()
                    best_radii = lp_r.copy()
        except Exception:
            continue
            
    # 4. Phase B: Joint center-radius SLSQP optimization
    bounds_joint = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-6, 0.5)] * n
    x0_joint = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_radii * 0.95])
    try:
        res_joint = minimize(lambda v: -np.sum(v[2*n:]), x0_joint, method='SLSQP',
                             bounds=bounds_joint,
                             constraints={'type': 'ineq', 'fun': cons_joint, 'args': (n,)},
                             options={'maxiter': 15000, 'ftol': 1e-13})
        if np.isfinite(res_joint.fun):
            c_new = np.column_stack((res_joint.x[:n], res_joint.x[n:2*n]))
            lp_r, lp_s = solve_lp_radii(c_new, n, idx_i, idx_j, A_lp_pairs)
            if lp_r is not None and lp_s > best_sum:
                best_sum = lp_s
                best_centers = c_new.copy()
                best_radii = lp_r.copy()
    except Exception:
        pass
        
    # 5. Phase C: Aggressive Hill Climbing on Centers evaluated via LP
    if best_centers is not None:
        T = 0.5
        for step in range(4000):
            T *= 0.9994
            step_sz = 0.012 * (1.0 + T)
            i = rng.integers(n)
            c_trial = best_centers.copy()
            c_trial[i] += rng.uniform(-step_sz, step_sz, 2)
            c_trial[i] = np.clip(c_trial[i], 1e-5, 1.0 - 1e-5)
            
            lp_r, lp_s = solve_lp_radii(c_trial, n, idx_i, idx_j, A_lp_pairs)
            if lp_r is not None:
                if lp_s > best_sum or rng.random() < np.exp((lp_s - best_sum) / max(T, 1e-8)):
                    best_sum = lp_s
                    best_centers = c_trial.copy()
                    best_radii = lp_r.copy()
                    
    # 6. Final LP refinement & Safety Scaling
    lp_r_final, lp_s_final = solve_lp_radii(best_centers, n, idx_i, idx_j, A_lp_pairs)
    if lp_r_final is not None:
        best_radii = lp_r_final
        best_sum = lp_s_final
        
    scale = 1.0
    wall = np.minimum(np.minimum(best_centers[:, 0], 1.0 - best_centers[:, 0]),
                      np.minimum(best_centers[:, 1], 1.0 - best_centers[:, 1]))
    scale = min(scale, np.min(wall / np.maximum(best_radii, 1e-12)))
    
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_pair = best_radii[:, np.newaxis] + best_radii[np.newaxis, :]
    scale = min(scale, np.min(dists[idx_i, idx_j] / np.maximum(r_pair[idx_i, idx_j], 1e-12)))
    
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    # Fallback guarantee
    if best_centers is None:
        grid = np.array([(0.1 + i*0.18, 0.1 + j*0.18) for j in range(5) for i in range(5)])
        best_centers = np.vstack([grid, [[0.55, 0.55]]])
        best_radii = np.full(26, 0.08)
        best_sum = float(np.sum(best_radii))
        
    return best_centers, best_radii, best_sum
