# sol_000067 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000026 (state f081a56f) state=62163a4d sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(v):
    """Minimize negative sum of radii."""
    return -np.sum(v[2*N_CIRCLES:])

def constraints(v):
    """Compute inequality constraints: boundaries and non-overlap."""
    n = N_CIRCLES
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    
    c = []
    # Boundary constraints
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise non-overlap constraints using squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    sum_r = r[:, np.newaxis] + r[np.newaxis, :]
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.append(dist_sq[mask] - sum_r[mask]**2)
    
    return np.concatenate(c)

def run_packing():
    n = N_CIRCLES
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_v = None
    
    # Multiple restarts with diverse initializations
    for seed in range(20):
        np.random.seed(seed)
        
        v0_list = []
        
        # 1. Perturbed Hexagonal Lattice
        r_init_val = 0.09
        centers_hex = []
        y = r_init_val
        row = 0
        while len(centers_hex) < n + 5:
            x = r_init_val + (row % 2) * r_init_val
            while x <= 1 - r_init_val and len(centers_hex) < n + 5:
                centers_hex.append([x, y])
                x += 2 * r_init_val
            y += np.sqrt(3) * r_init_val
            row += 1
            
        pts_hex = np.array(centers_hex[:n])
        pts_hex += np.random.uniform(-0.005, 0.005, size=pts_hex.shape)
        pts_hex = np.clip(pts_hex, 0.02, 0.98)
        v0_list.append(np.concatenate([pts_hex[:, 0], pts_hex[:, 1], np.full(n, 0.02)]))
        
        # 2. Random valid start
        pts_rand = np.random.uniform(0.05, 0.95, size=(n, 2))
        v0_list.append(np.concatenate([pts_rand[:, 0], pts_rand[:, 1], np.full(n, 0.02)]))
        
        # 3. Perturbed Grid
        pts_grid = []
        for i in range(6):
            for j in range(5):
                if len(pts_grid) < n:
                    pts_grid.append([0.1 + i*0.16, 0.1 + j*0.18])
        pts_grid = np.array(pts_grid)
        pts_grid += np.random.uniform(-0.01, 0.01, size=pts_grid.shape)
        pts_grid = np.clip(pts_grid, 0.02, 0.98)
        v0_list.append(np.concatenate([pts_grid[:, 0], pts_grid[:, 1], np.full(n, 0.02)]))
        
        for v0 in v0_list:
            try:
                res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                cur_sum = -res.fun
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_v = res.x.copy()
            except Exception:
                pass
                
    if best_v is None:
        return np.zeros((n, 2)), np.zeros(n), 0.0
        
    x_sol = best_v[:n]
    y_sol = best_v[n:2*n]
    r_sol = best_v[2*n:]
    centers_sol = np.column_stack((x_sol, y_sol))
    
    # Post-processing to guarantee strict validity per validation rules
    for _ in range(20):
        changed = False
        for i in range(n):
            limit = min(centers_sol[i, 0], 1.0 - centers_sol[i, 0],
                        centers_sol[i, 1], 1.0 - centers_sol[i, 1])
            for j in range(n):
                if i == j: continue
                d = np.hypot(centers_sol[i, 0] - centers_sol[j, 0],
                             centers_sol[i, 1] - centers_sol[j, 1])
                if d < limit + r_sol[j]:
                    limit = d - r_sol[j]
            if r_sol[i] > limit + 1e-10:
                r_sol[i] = max(limit, 0.0)
                changed = True
        if not changed:
            break
            
    # Final safety scale if numerical drift occurred
    min_slack = 1.0
    for i in range(n):
        min_slack = min(min_slack, centers_sol[i, 0] - r_sol[i])
        min_slack = min(min_slack, 1.0 - centers_sol[i, 0] - r_sol[i])
        min_slack = min(min_slack, centers_sol[i, 1] - r_sol[i])
        min_slack = min(min_slack, 1.0 - centers_sol[i, 1] - r_sol[i])
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(centers_sol[i, 0] - centers_sol[j, 0],
                         centers_sol[i, 1] - centers_sol[j, 1])
            min_slack = min(min_slack, d - r_sol[i] - r_sol[j])
            
    if min_slack < -1e-11:
        scale = 1.0 + min_slack / np.max(r_sol)
        r_sol *= max(scale, 0.99999)
        
    return centers_sol, r_sol, float(np.sum(r_sol))
