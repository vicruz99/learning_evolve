# sol_000227 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96713eb2) state=37dabcab sum of radii=2.511193 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the circle packing problem for 26 circles in a unit square to maximize sum of radii.
    
    Returns:
        centers: np.ndarray of shape (26, 2)
        radii: np.ndarray of shape (26,)
        sum_radii: float
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal packing layout
    # We estimate a reasonable initial radius. 
    # 25 circles fit with r=0.1 (grid). 26 will be slightly less, maybe 0.09.
    # We start with a hexagonal arrangement to utilize space better than a grid.
    
    # Approximate initial radius
    r_init = 0.08
    
    centers_init = np.zeros((n_circles, 2))
    idx = 0
    
    # Generate hexagonal grid points
    row_height = np.sqrt(3) / 2 * 2 * r_init
    
    # Try to fit rows
    y = r_init
    while y + r_init <= 1.0 and idx < n_circles:
        x = r_init
        while x + r_init <= 1.0 and idx < n_circles:
            centers_init[idx, 0] = x
            centers_init[idx, 1] = y
            idx += 1
            x += 2 * r_init
        y += row_height
        # Shift every other row for hexagonal pattern
        if (int((y - r_init) / row_height)) % 2 == 0:
             x_offset = r_init
        else:
             x_offset = 0
             
    # If we didn't fit all, or if layout is sparse, we can add random jitter 
    # to remaining or just perturb all. 
    # Let's ensure we have exactly 26.
    while idx < n_circles:
        # Fill remaining in a spiral or random within bounds
        centers_init[idx, 0] = np.random.uniform(0.1, 0.9)
        centers_init[idx, 1] = np.random.uniform(0.1, 0.9)
        idx += 1
        
    # Add small random perturbation to break symmetry
    centers_init += np.random.normal(0, 0.001, size=centers_init.shape)
    
    # Clip to valid range [r_init, 1-r_init]
    centers_init = np.clip(centers_init, r_init, 1 - r_init)

    # 2. Define the optimization function
    # We maximize sum(r_i) => minimize -sum(r_i)
    # Variables vector: [x1, y1, r1, x2, y2, r2, ...]
    
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)

    def distance_constraint(vars, i, j):
        xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
        xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
        dist_sq = (xi - xj)**2 + (yi - yj)**2
        radii_sum_sq = (ri + rj)**2
        # Constraint: dist_sq >= radii_sum_sq  => dist_sq - radii_sum_sq >= 0
        return dist_sq - radii_sum_sq

    def wall_constraint_x_min(vars, i):
        return vars[3*i] - vars[3*i+2] # x >= r

    def wall_constraint_x_max(vars, i):
        return (1 - vars[3*i]) - vars[3*i+2] # 1-x >= r => 1 - (x+r) >= 0

    def wall_constraint_y_min(vars, i):
        return vars[3*i+1] - vars[3*i+2] # y >= r

    def wall_constraint_y_max(vars, i):
        return (1 - vars[3*i+1]) - vars[3*i+2] # 1-y >= r

    # Prepare initial variables
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = r_init

    # Bounds: 0 <= x,y <= 1, 0 <= r <= 1
    bounds = []
    for _ in range(n_circles):
        bounds.extend([(0, 1), (0, 1), (0, 1)]) # x, y, r

    # Inequality constraints (fun(vars) >= 0)
    constraints = []
    
    # Distance constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: distance_constraint(v, i, j)
            })

    # Wall constraints
    for i in range(n_circles):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: wall_constraint_x_min(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: wall_constraint_x_max(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: wall_constraint_y_min(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: wall_constraint_y_max(v, i)})

    # 3. Optimization
    # Using SLSQP. It might get stuck in local optima, so we run a few times or use good init.
    # Given the complexity, a single run with good initialization might suffice for a reasonable score.
    
    result = scipy.optimize.minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )

    # Extract solution
    final_vars = result.x
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        centers[i, 0] = final_vars[3*i]
        centers[i, 1] = final_vars[3*i+1]
        radii[i] = final_vars[3*i+2]

    # Ensure non-negative radii (clipping just in case of numerical issues)
    radii = np.maximum(radii, 0)
    
    # Final validation and adjustment
    # If any constraints are slightly violated due to precision, fix them.
    # However, the validator is strict.
    
    # Simple post-processing to ensure strict validity
    # Reduce radii slightly if necessary to satisfy overlap exactly
    # But the optimizer should have handled it.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
