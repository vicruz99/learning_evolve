# sol_000085 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000046 (state 0aa7241c) state=b486c7b4 sum of radii=1.877221 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_initial_configs(n):
    """Generates multiple high-quality initial layouts based on hexagonal lattices."""
    configs = []
    row_counts_list = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [5, 5, 6, 5, 5],
        [4, 6, 6, 5, 5],
        [6, 6, 5, 4, 5]
    ]
    t0 = 0.10
    
    for rc in row_counts_list:
        pts = []
        y = t0
        row = 0
        for count in rc:
            shift = t0 if row % 2 == 1 else 0.0
            x = t0 + shift
            for _ in range(count):
                if len(pts) < n:
                    pts.append([x, y])
                x += 2 * t0
            y += np.sqrt(3) * t0
            row += 1
            
        base_pts = np.array(pts[:n])
        configs.append(base_pts)
        
        # Generate shifted and scaled variants to break symmetry
        shifts = [(0.0, 0.0), (t0, 0.0), (0.0, np.sqrt(3)/2 * t0), (t0/2, np.sqrt(3)/4 * t0)]
        for sx, sy in shifts:
            p = base_pts + np.array([sx, sy])
            p_min = p.min(axis=0)
            p_max = p.max(axis=0)
            p_range = p_max - p_min + 1e-9
            p = (p - p_min) / p_range * 0.8 + 0.1
            configs.append(p)
            
    # Add perturbed versions to escape local minima
    np.random.seed(42)
    for _ in range(5):
        p = configs[0].copy()
        p += np.random.uniform(-0.02, 0.02, p.shape)
        p = np.clip(p, 0.05, 0.95)
        configs.append(p)
        
    return configs

def objective_equal(vars, n):
    """Objective: minimize negative t => maximize t"""
    return -vars[-1]

def constraints_equal(vars, n):
    """Constraints: boundary distances >= t, pairwise squared distances >= 4*t^2"""
    t = vars[-1]
    c = vars[:-1].reshape(n, 2)
    cx = c[:, 0]
    cy = c[:, 1]
    
    # Pairwise squared distances
    cx_d = cx[:, None] - cx[None, :]
    cy_d = cy[:, None] - cy[None, :]
    d_sq = cx_d**2 + cy_d**2
    idx = np.triu_indices(n, k=1)
    
    con = np.concatenate([
        cx - t,                  # x - t >= 0
        1.0 - cx - t,           # 1 - x - t >= 0
        cy - t,                  # y - t >= 0
        1.0 - cy - t,           # 1 - y - t >= 0
        d_sq[idx] - 4.0 * t**2  # dist^2 - 4t^2 >= 0
    ])
    return con

def run_packing():
    n = 26
    best_t = 0.0
    best_centers = None
    
    # Bounds: centers in [0,1], radius t in [0.05, 0.12]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.05, 0.12)]
    cons = {'type': 'ineq', 'fun': constraints_equal, 'args': (n,)}
    
    init_configs = compute_initial_configs(n)
    
    # Optimize from each configuration
    for cfg in init_configs:
        # Start with a feasible small t to ensure initial constraint satisfaction
        x0 = np.concatenate([cfg.flatten(), [0.05]])
        try:
            res = minimize(objective_equal, x0, args=(n,), method='SLSQP', 
                          bounds=bounds, constraints=cons, 
                          options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if res.x[-1] > best_t:
                best_t = res.x[-1]
                best_centers = res.x[:-1].reshape(n, 2)
        except Exception:
            continue
            
    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = init_configs[0]
        best_t = 0.09
        
    # Phase 2: Extract exact maximal feasible radii for the optimized centers.
    # This allows circles in less constrained regions to grow, often increasing the sum.
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundaries
        min_d = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                    best_centers[i, 1], 1.0 - best_centers[i, 1])
        # Distance to other circles
        for j in range(n):
            if i != j:
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                if d < min_d:
                    min_d = d
        radii[i] = min_d / 2.0
        
    # Apply a tiny safety margin to strictly satisfy the 1e-12 tolerance in validation
    radii *= 0.99999
    sum_radii = float(np.sum(radii))
    
    return best_centers, radii, sum_radii
