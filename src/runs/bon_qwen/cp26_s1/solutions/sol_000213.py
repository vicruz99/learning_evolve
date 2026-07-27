# sol_000213 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cccf4974) state=148148cd sum of radii=2.615283 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars, n):
    """Returns an array of constraint values that must be >= 0."""
    centers = vars[:2 * n].reshape(n, 2)
    radii = vars[2 * n:]
    
    c = []
    # Wall constraints: center ± radius within [0, 1]
    for i in range(n):
        c.append(centers[i, 0] - radii[i])
        c.append(1.0 - centers[i, 0] - radii[i])
        c.append(centers[i, 1] - radii[i])
        c.append(1.0 - centers[i, 1] - radii[i])
        
    # Overlap constraints: distance between centers >= sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.hypot(dx, dy)
            c.append(dist - radii[i] - radii[j])
            
    return np.array(c)

def objective_func(vars, n):
    """Objective: maximize sum of radii (minimize negative sum)."""
    radii = vars[2 * n:]
    return -np.sum(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: 5x5 grid + 1 circle in center
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
    centers.append([0.5, 0.5])
    centers = np.array(centers)
    radii = np.ones(n) * 0.1
    
    # Break symmetry to help escape local minima
    np.random.seed(42)
    centers += np.random.uniform(-0.008, 0.008, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Flatten to optimization vector: [x0, y0, ..., xN-1, yN-1, r0, ..., rN-1]
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds for variables
    bounds = [(0.0, 1.0)] * (2 * n)  # centers in [0, 1]
    bounds += [(1e-5, 0.5)] * n      # radii positive and reasonable upper bound
    
    # Define constraints for SLSQP
    cons = {
        'type': 'ineq', 
        'fun': compute_constraints, 
        'args': (n,)
    }
    
    # 2. Optimization
    res = minimize(
        objective_func, 
        x0, 
        args=(n,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
    )
    
    # 3. Extract and format results
    centers_opt = res.x[:2 * n].reshape(n, 2)
    radii_opt = res.x[2 * n:]
    
    # 4. Strict feasibility post-processing
    # Ensure no circle crosses the boundary
    for i in range(n):
        margin_x = min(centers_opt[i, 0], 1.0 - centers_opt[i, 0])
        margin_y = min(centers_opt[i, 1], 1.0 - centers_opt[i, 1])
        max_r_wall = min(margin_x, margin_y) - 1e-9
        radii_opt[i] = min(radii_opt[i], max_r_wall)
        
    # Iteratively resolve any numerical overlaps
    for _ in range(30):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers_opt[i, 0] - centers_opt[j, 0], 
                             centers_opt[i, 1] - centers_opt[j, 1])
                r_sum = radii_opt[i] + radii_opt[j]
                if d < r_sum - 1e-9:
                    shrink = (r_sum - d) / 2.0 + 1e-9
                    radii_opt[i] -= shrink
                    radii_opt[j] -= shrink
                    overlap_found = True
        if not overlap_found:
            break
            
    # Ensure strictly positive radii
    radii_opt = np.maximum(radii_opt, 1e-6)
    
    sum_radii = np.sum(radii_opt)
    return centers_opt, radii_opt, sum_radii
