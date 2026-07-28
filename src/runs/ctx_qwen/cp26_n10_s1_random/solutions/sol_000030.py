# sol_000030 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state bde5dee5) state=6df322f9 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def objective_func(params):
    n = 26
    r = params[2::3]
    return -np.sum(r)

def constraint_func(params):
    n = 26
    cx = params[0::3]
    cy = params[1::3]
    r = params[2::3]
    
    # Preallocate constraint array: 4 boundary constraints per circle + pairwise overlaps
    n_cons = n * 4 + n * (n - 1) // 2
    cons = np.empty(n_cons)
    idx = 0
    
    for i in range(n):
        cons[idx] = cx[i] - r[i]
        idx += 1
        cons[idx] = 1.0 - cx[i] - r[i]
        idx += 1
        cons[idx] = cy[i] - r[i]
        idx += 1
        cons[idx] = 1.0 - cy[i] - r[i]
        idx += 1
        
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            dist_sq = dx*dx + dy*dy
            r_sum = r[i] + r[j]
            cons[idx] = dist_sq - r_sum*r_sum
            idx += 1
            
    return cons

def solve_radii_lp(centers):
    n = centers.shape[0]
    cx, cy = centers[:, 0], centers[:, 1]
    
    # Distance to nearest boundary limits radius
    wall_dists = np.minimum(np.minimum(cx, 1 - cx), np.minimum(cy, 1 - cy))
    bounds = [(0.0, max(w, 1e-9)) for w in wall_dists]
    
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    
    # Vectorized pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    # Maximize sum of radii -> minimize negative sum
    c = -np.ones(n)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x, np.sum(res.x)
    
    # Fallback if LP fails (rare)
    safe_r = np.minimum(wall_dists, np.min(dists, axis=1) / 2.0)
    return np.maximum(safe_r, 1e-6), np.sum(np.maximum(safe_r, 1e-6))

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    base_r = 0.10
    configs = []
    
    # Generate multiple hexagonal initial configurations with perturbations
    for seed in range(5):
        np.random.seed(seed * 17 + 42)
        pts = []
        for row in range(7):
            for col in range(5):
                if len(pts) >= n: break
                x = col * 2 * base_r + (row % 2) * base_r + base_r
                y = row * np.sqrt(3) * base_r + base_r
                pts.append([x, y])
        pts = np.array(pts[:n])
        
        # Normalize to fit within [0,1] with a margin
        pts -= pts.min(axis=0)
        pts /= pts.max(axis=0)
        pts = pts * 0.85 + 0.075
        
        # Add controlled noise to break symmetry
        noise = np.random.uniform(-0.03, 0.03, pts.shape)
        pts = np.clip(pts + noise, 0.1, 0.9)
        
        params = np.empty(3 * n)
        params[0::3] = pts[:, 0]
        params[1::3] = pts[:, 1]
        params[2::3] = base_r * 0.95
        configs.append(params)
        
    best_sum = 0.0
    best_centers = None
    
    # Bounds for optimization: x,y in [0,1], r in [1e-6, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    
    for x0 in configs:
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraint_func},
                           options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            
            if np.isfinite(res.fun):
                cx = res.x[0::3]
                cy = res.x[1::3]
                r = res.x[2::3]
                
                centers = np.column_stack((cx, cy))
                # Verify constraints are satisfied (within numerical tolerance)
                cons_val = constraint_func(res.x)
                if np.all(cons_val >= -1e-5):
                    s = np.sum(r)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers.copy()
        except Exception:
            continue
            
    # Fallback if optimization yields no valid result
    if best_centers is None:
        x0 = configs[0]
        best_centers = np.column_stack((x0[0::3], x0[1::3]))
        
    # Ensure centers are strictly inside the square
    best_centers = np.clip(best_centers, 1e-7, 1.0 - 1e-7)
    
    # Final exact radius optimization via LP for the best center layout
    final_radii, final_sum = solve_radii_lp(best_centers)
    
    return best_centers, final_radii, float(final_sum)
