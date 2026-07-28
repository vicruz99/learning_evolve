# sol_000219 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000216 (state 64a1292d) state=999454df sum of radii=1.762369 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_lp_matrix(n):
    """Precompute the constant A_ub matrix structure for pairwise constraints."""
    m = n * (n - 1) // 2
    A = np.zeros((m, n))
    i, j = np.triu_indices(n, k=1)
    for k, (ii, jj) in enumerate(zip(i, j)):
        A[k, ii] = 1.0
        A[k, jj] = 1.0
    return A, i, j

def solve_lp_radii(centers, A, i, j):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    bounds = [(0.0, max(l, 1e-9)) for l in limits]
    
    diffs = centers[i] - centers[j]
    b_ub = np.sqrt(np.sum(diffs**2, axis=1))
    
    res = linprog(-np.ones(n), A_ub=A, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x, -res.fun
    return None, 0.0

def get_constraints(x, n):
    """Inequality constraints >= 0 for SLSQP."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    cons = []
    # Boundary
    cons.append(cx - r)
    cons.append(1.0 - cx - r)
    cons.append(cy - r)
    cons.append(1.0 - cy - r)
    
    # Pairwise non-overlap
    cx_d = cx[:, None] - cx[None, :]
    cy_d = cy[:, None] - cy[None, :]
    dists = np.sqrt(cx_d**2 + cy_d**2)
    r_sum = r[:, None] + r[None, :]
    i, j = np.triu_indices(n, k=1)
    cons.append(dists[i, j] - r_sum[i, j])
    
    return np.concatenate(cons)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    A, i, j = compute_lp_matrix(n)
    
    # 1. Generate diverse initial configurations
    configs = []
    patterns = [
        [5,6,5,6,4], [6,5,6,5,4], [5,5,6,5,5], [4,6,6,6,4], 
        [6,6,5,5,4], [5,4,6,6,5], [5,6,4,6,5], [6,5,5,6,4],
        [5,5,5,5,6], [6,6,4,5,5], [5,6,6,5,4], [4,5,6,5,6]
    ]
    for pat in patterns:
        if sum(pat) < n: continue
        r0 = 0.10
        pts = []
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) < n: pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
        configs.append(np.array(pts[:n]))
        
    for _ in range(12):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    bounds_vars = [(0.0, 1.0)] * (2*n) + [(1e-6, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': lambda x: get_constraints(x, n)}
    
    # 2. SLSQP Joint Optimization + LP Refinement
    for cfg in configs:
        c = cfg + rng.uniform(-0.02, 0.02, cfg.shape)
        c = np.clip(c, 0.05, 0.95)
        
        x0 = np.zeros(3*n)
        x0[0::3] = c[:, 0]
        x0[1::3] = c[:, 1]
        x0[2::3] = 0.09
        
        try:
            res = minimize(lambda x: -np.sum(x[2*n:]), x0, method='SLSQP',
                           bounds=bounds_vars, constraints=cons_dict,
                           options={'maxiter': 2500, 'ftol': 1e-12})
            if not np.isfinite(res.fun): continue
            
            cx_opt = res.x[0::3]
            cy_opt = res.x[1::3]
            c_opt = np.column_stack((cx_opt, cy_opt))
            
            # LP refinement on polished centers
            r_lp, s_lp = solve_lp_radii(c_opt, A, i, j)
            if r_lp is None: continue
            
            # Quick validity check
            valid = True
            if np.any(r_lp < 0): valid = False
            else:
                for k in range(n):
                    if c_opt[k, 0] < r_lp[k] - 1e-8 or c_opt[k, 0] > 1 - r_lp[k] + 1e-8:
                        valid = False; break
                    if c_opt[k, 1] < r_lp[k] - 1e-8 or c_opt[k, 1] > 1 - r_lp[k] + 1e-8:
                        valid = False; break
                if valid:
                    for k in range(n):
                        for l in range(k+1, n):
                            dd = np.hypot(c_opt[k,0]-c_opt[l,0], c_opt[k,1]-c_opt[l,1])
                            if dd < r_lp[k] + r_lp[l] - 1e-8:
                                valid = False; break
                        if not valid: break
            
            if valid and s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_opt.copy()
                best_radii = r_lp.copy()
                
        except Exception:
            continue
            
    # 3. Iterative Local Search (Coordinate Ascent + LP)
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        for _ in range(80):
            improved = False
            order = rng.permutation(n)
            for k in order:
                old_c = curr_c[k].copy()
                # Try several random perturbations
                for _ in range(6):
                    step = 0.012
                    new_c = curr_c[k] + rng.uniform(-step, step, 2)
                    new_c = np.clip(new_c, 0.02, 0.98)
                    curr_c[k] = new_c
                    
                    limits = np.minimum(np.minimum(curr_c[:, 0], 1.0 - curr_c[:, 0]),
                                        np.minimum(curr_c[:, 1], 1.0 - curr_c[:, 1]))
                    bds = [(0.0, max(l, 1e-9)) for l in limits]
                    diffs = curr_c[i] - curr_c[j]
                    b_ub = np.sqrt(np.sum(diffs**2, axis=1))
                    
                    lp_res = linprog(-np.ones(n), A_ub=A, b_ub=b_ub, bounds=bds, method='highs')
                    if lp_res.success:
                        new_s = -lp_res.fun
                        if new_s > curr_s + 1e-8:
                            curr_s = new_s
                            curr_c[k] = new_c
                            curr_r = lp_res.x.copy()
                            improved = True
                            break
                    curr_c[k] = old_c
                if improved:
                    break
            if not improved:
                break
                
        best_centers = curr_c
        best_radii = curr_r
        best_sum = curr_s
        
    # 4. Strict Safety Scaling
    if best_centers is not None:
        scale = 1.0
        for k in range(n):
            x, y, r = best_centers[k, 0], best_centers[k, 1], best_radii[k]
            if r > 1e-12:
                scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
                
        dists_mat = np.sqrt(np.sum((best_centers[:, None, :] - best_centers[None, :, :])**2, axis=2))
        for p in range(n):
            for q in range(p+1, n):
                d = dists_mat[p, q]
                rs = best_radii[p] + best_radii[q]
                if rs > 1e-12:
                    scale = min(scale, d / rs)
                    
        best_radii *= scale * 0.9999999
        best_sum = float(np.sum(best_radii))
    else:
        best_centers = np.ones((n, 2)) * 0.5
        best_radii = np.full(n, 0.01)
        best_sum = float(np.sum(best_radii))
        
    return best_centers, best_radii, best_sum
