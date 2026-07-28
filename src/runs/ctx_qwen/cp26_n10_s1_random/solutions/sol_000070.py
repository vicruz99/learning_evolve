# sol_000070 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000052 (state e51e4326) state=545a1393 sum of radii=2.139170 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective: minimize negative radius t (equivalent to maximizing t)"""
    return -vars[-1]

def constraints(vars, n):
    """
    Computes inequality constraints >= 0:
    1. Boundary: x >= t, 1-x >= t, y >= t, 1-y >= t
    2. Non-overlap: ||c_i - c_j||^2 >= 4*t^2
    """
    xs = vars[0::2]
    ys = vars[1::2]
    t = vars[-1]
    
    # Preallocate constraint array
    con = np.empty(4*n + n*(n-1)//2)
    idx = 0
    
    # Boundary constraints
    con[idx:idx+n] = xs - t; idx += n
    con[idx:idx+n] = 1.0 - xs - t; idx += n
    con[idx:idx+n] = ys - t; idx += n
    con[idx:idx+n] = 1.0 - ys - t; idx += n
    
    # Pairwise squared distance constraints
    xd = xs[:, np.newaxis] - xs[np.newaxis, :]
    yd = ys[:, np.newaxis] - ys[np.newaxis, :]
    dist_sq = xd**2 + yd**2
    np.fill_diagonal(dist_sq, np.inf)
    
    # Extract upper triangle (i < j) and enforce dist_sq >= 4t^2
    con[idx:] = dist_sq[np.triu_indices(n, k=1)] - 4.0 * t * t
    return con

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_t = 0.0
    
    configs = []
    
    # 1. Hexagonal lattice initialization
    pts = []
    r0 = 0.095
    dy = np.sqrt(3) * r0
    y = r0
    row = 0
    while len(pts) < n:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r0
        y += dy
        row += 1
    hex_cfg = np.array(pts[:n])
    configs.append(hex_cfg)
    
    # 2. Perturbed hexagonal configurations to escape local minima
    np.random.seed(42)
    for sigma in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        cfg = hex_cfg + np.random.normal(0, sigma, hex_cfg.shape)
        cfg = np.clip(cfg, 0.05, 0.95)
        configs.append(cfg)
        
    # 3. Regular grid fallback
    grid = []
    step = 0.2
    for i in range(5):
        for j in range(5):
            grid.append([0.1 + i * step, 0.1 + j * step])
    grid.append([0.5, 0.5])
    configs.append(np.array(grid[:n]))
    
    # Bounds: centers in [0, 1], radius t in [0.05, 0.12]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.05, 0.12)]
    cons = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    # Multi-start optimization
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), [0.095]])
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False}
            )
            t_val = res.x[-1]
            if t_val > best_t:
                # Verify constraints are satisfied within tolerance
                c_vals = constraints(res.x, n)
                if np.min(c_vals) >= -1e-6:
                    best_t = t_val
                    best_centers = np.column_stack((res.x[0::2], res.x[1::2])).copy()
                    best_sum = t_val * n
        except Exception:
            continue
            
    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = configs[0]
        best_t = 0.09
        
    # Final exact geometric clearance check to guarantee strict validity
    cx, cy = best_centers[:, 0], best_centers[:, 1]
    min_wall = min(np.min(cx), np.min(1.0 - cx), np.min(cy), np.min(1.0 - cy))
    
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    feasible_t = min(min_wall, min_pair)
    if feasible_t < best_t:
        best_t = feasible_t * 0.999999
        
    radii = np.full(n, best_t)
    return best_centers, radii, float(np.sum(radii))
