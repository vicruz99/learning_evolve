# sol_000065 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000026 (state f081a56f) state=a4d312af sum of radii=2.620609 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v, n):
    """Objective function to minimize: negative sum of radii."""
    return -np.sum(v[2*n:])

def constraints_fun(v, n):
    """Constraint function returning values that must be >= 0."""
    xs = v[:n]
    ys = v[n:2*n]
    rs = v[2*n:]
    
    # Boundary constraints
    cons = [
        xs - rs,           # x >= r
        1 - xs - rs,       # x + r <= 1
        ys - rs,           # y >= r
        1 - ys - rs        # y + r <= 1
    ]
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_sq = dx**2 + dy**2
    
    sum_r = rs[:, None] + rs[None, :]
    sum_r_sq = sum_r**2
    
    # Extract upper triangle to avoid duplicate constraints and self-constraints
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    cons.append((dist_sq - sum_r_sq)[mask])
    
    return np.concatenate(cons)

def run_packing():
    n = 26
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints_fun, 'args': (n,)}
    
    best_sum = -1.0
    best_v = None
    
    # Generate diverse initial configurations to escape local minima
    configs = []
    
    # 1. Standard Hexagonal Lattice
    r_hex = 0.10
    y, row = r_hex, 0
    pts = []
    while len(pts) < n + 15:
        x_start = r_hex + (row % 2) * r_hex
        x = x_start
        while x <= 1 - r_hex:
            pts.append([x, y])
            x += 2 * r_hex
        y += np.sqrt(3) * r_hex
        row += 1
    configs.append(np.array(pts[:n]))
    
    # 2. Compressed Hexagonal Lattice
    r_hex2 = 0.09
    y, row = r_hex2, 0
    pts2 = []
    while len(pts2) < n + 15:
        x_start = r_hex2 + (row % 2) * r_hex2 * 0.9
        x = x_start
        while x <= 1 - r_hex2:
            pts2.append([x, y])
            x += 2 * r_hex2
        y += np.sqrt(3) * r_hex2 * 0.95
        row += 1
    configs.append(np.array(pts2[:n]))
    
    # 3 & 4. Random Valid Packings
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        for _ in range(50):
            pts3 = np.random.uniform(0.05, 0.95, size=(n, 2))
            dx3 = pts3[:, 0, None] - pts3[None, :, 0]
            dy3 = pts3[:, 1, None] - pts3[None, :, 1]
            dists3 = np.sqrt(dx3**2 + dy3**2)
            np.fill_diagonal(dists3, np.inf)
            if np.min(dists3) > 0.10:
                configs.append(pts3)
                break
                
    # 5. Perturbed Grid
    np.random.seed(777)
    grid = np.zeros((n, 2))
    for i in range(n):
        c = i % 6
        r_idx = i // 6
        grid[i] = [0.1 + c * 0.16, 0.1 + r_idx * 0.16]
    grid += np.random.uniform(-0.015, 0.015, size=grid.shape)
    grid = np.clip(grid, 0.05, 0.95)
    configs.append(grid)
    
    # Optimize each configuration
    for centers_init in configs:
        dx = centers_init[:, 0, None] - centers_init[None, :, 0]
        dy = centers_init[:, 1, None] - centers_init[None, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        # Initialize radii slightly smaller than half the minimum distance
        r_init = np.full(n, min(min_d / 2 * 0.98, 0.12))
        v0 = np.concatenate([centers_init[:, 0], centers_init[:, 1], r_init])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds, args=(n,),
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is not None:
        xs = best_v[:n]
        ys = best_v[n:2*n]
        rs = best_v[2*n:]
        
        # Enforce non-negative radii
        rs = np.maximum(rs, 0.0)
        
        # Enforce boundary constraints strictly
        rs = np.minimum(rs, np.minimum(np.minimum(xs, 1-xs), np.minimum(ys, 1-ys)))
        
        # Iteratively resolve any minor overlaps due to numerical precision
        for _ in range(10):
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                    if rs[i] + rs[j] > dist - 1e-9:
                        shrink = (rs[i] + rs[j] - dist + 1e-8) / 2.0
                        rs[i] -= shrink
                        rs[j] -= shrink
                        rs[i] = max(rs[i], 0.0)
                        rs[j] = max(rs[j], 0.0)
                        changed = True
            if not changed:
                break
                
        centers = np.column_stack((xs, ys))
        return centers, rs, float(np.sum(rs))
        
    # Fallback configuration
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.08)
    for i in range(n):
        centers[i] = [0.08 + (i%6)*0.16, 0.08 + (i//6)*0.16]
    return centers, radii, 26*0.08
