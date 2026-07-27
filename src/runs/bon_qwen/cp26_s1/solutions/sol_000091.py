# sol_000091 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9fb5006a) state=1c253c34 sum of radii=2.504579 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # --- Initialization ---
    # We start with a valid configuration of small circles arranged in a hexagonal grid.
    # This topology is dense and likely close to the optimal arrangement.
    # We use a reference radius for spacing to ensure circles are well-separated initially,
    # and a smaller actual radius to satisfy constraints strictly.
    
    r_init = 0.02
    ref_r = 0.04 # Reference radius for determining grid spacing
    
    h_dist = 2 * ref_r
    v_dist = math.sqrt(3) * ref_r
    
    centers_init = []
    
    # Pattern to fit 26 circles: 5 rows with counts 6, 5, 5, 5, 5
    row_counts = [6, 5, 5, 5, 5]
    
    current_y = ref_r # Start y position based on ref_r margin
    for i, count in enumerate(row_counts):
        # Shift odd rows horizontally for hexagonal packing
        x_start = ref_r + (i % 2) * ref_r
        
        for j in range(count):
            cx = x_start + j * h_dist
            cy = current_y
            centers_init.append([cx, cy])
        
        current_y += v_dist
        
    centers_init = np.array(centers_init[:n])
    
    # Center the configuration in the unit square [0,1]x[0,1]
    min_x, max_x = np.min(centers_init[:, 0]), np.max(centers_init[:, 0])
    min_y, max_y = np.min(centers_init[:, 1]), np.max(centers_init[:, 1])
    
    shift_x = 0.5 - (min_x + max_x) / 2
    shift_y = 0.5 - (min_y + max_y) / 2
    
    centers_init[:, 0] += shift_x
    centers_init[:, 1] += shift_y
    
    # --- Optimization Setup ---
    # Variables: [x1...xn, y1...yn, r1...rn]
    # Total variables = 3 * 26 = 78
    x0 = np.zeros(3 * n)
    x0[0:n] = centers_init[:, 0]
    x0[n:2*n] = centers_init[:, 1]
    x0[2*n:3*n] = r_init * np.ones(n)
    
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x bounds
        bounds.append((0.0, 1.0)) # y bounds
        bounds.append((1e-6, 0.5)) # r bounds (must be positive)
    
    # Constraint functions
    
    # 1. Boundary constraints: circles must be inside [0,1]x[0,1]
    #    x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    def constr_boundary(x_vars):
        xs = x_vars[0:n]
        ys = x_vars[n:2*n]
        rs = x_vars[2*n:3*n]
        return np.concatenate([
            xs - rs,
            1.0 - xs - rs,
            ys - rs,
            1.0 - ys - rs
        ])

    # 2. Non-overlap constraints: distance between centers >= sum of radii
    #    (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    def constr_no_overlap(x_vars):
        xs = x_vars[0:n]
        ys = x_vars[n:2*n]
        rs = x_vars[2*n:3*n]
        
        # Vectorized pairwise distance calculation
        # Create matrices for broadcasting
        dx = xs[:, np.newaxis] - xs[np.newaxis, :]
        dy = ys[:, np.newaxis] - ys[np.newaxis, :]
        dr = rs[:, np.newaxis] + rs[np.newaxis, :]
        
        dist_sq = dx**2 + dy**2
        rad_sum_sq = dr**2
        
        diff = dist_sq - rad_sum_sq
        
        # Return only upper triangle constraints (i < j) to avoid redundancy
        idx = np.triu_indices(n, k=1)
        return diff[idx]

    constraints = [
        {'type': 'ineq', 'fun': constr_boundary},
        {'type': 'ineq', 'fun': constr_no_overlap}
    ]

    # Objective: Maximize sum of radii -> Minimize negative sum
    def objective(x_vars):
        return -np.sum(x_vars[2*n:3*n])

    # --- Run Optimizer ---
    try:
        # SLSQP is a robust solver for constrained non-linear problems
        # We allow sufficient iterations to converge
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 3000, 'ftol': 1e-9, 'disp': False})
        final_x = res.x
    except Exception:
        # Fallback to initial configuration if optimization fails
        final_x = x0

    best_centers = np.column_stack((final_x[0:n], final_x[n:2*n]))
    best_radii = final_x[2*n:3*n]
    
    # Safety clamp for radii to ensure non-negative
    best_radii = np.maximum(best_radii, 1e-9)
    
    # Final check: ensure circles are within bounds (fix potential numerical slips)
    # If a circle is slightly outside, shrink its radius to fit.
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        margin = min(x, 1 - x, y, 1 - y)
        if r > margin:
            best_radii[i] = max(0, margin)
        
    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, sum_radii
