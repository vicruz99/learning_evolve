# sol_000161 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3daa574a) state=04841438 sum of radii=2.471015 correctness=1.0
# stdout(first 200): `gtol` termination condition is satisfied. Number of iterations: 120, function evaluations: 9006, CG iterations: 276, optimality: 8.27e-09, constraint violation: 0.00e+00, execution time: 1.6e+02 s.
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def run_packing():
    n = 26
    
    # Initial placement: Grid pattern with small radius
    # 6 columns, 5 rows = 30 slots, we take first 26
    # This fits well in [0,1]x[0,1] with r=0.05
    r_init = 0.05
    cols = 6
    rows = 5
    
    centers = np.zeros((n, 2))
    radii = np.full(n, r_init)
    
    for i in range(n):
        col = i % cols
        row = i // cols
        
        # Spacing of 0.17 between centers to allow expansion
        # Range [0.1, 0.9] roughly
        x = 0.1 + col * 0.15
        y = 0.1 + row * 0.15
        
        # Clamp to valid range [r, 1-r]
        x = np.clip(x, r_init, 1 - r_init)
        y = np.clip(y, r_init, 1 - r_init)
        
        centers[i] = [x, y]

    # Flatten variables for optimization: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3 * i] = centers[i, 0]
        x0[3 * i + 1] = centers[i, 1]
        x0[3 * i + 2] = radii[i]

    # Define objective: Maximize sum of radii -> Minimize negative sum
    def objective(vars_vec):
        r_sum = 0
        for i in range(n):
            r_sum += vars_vec[3 * i + 2]
        return -r_sum

    # Define constraints
    # 1. Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    # 2. Non-overlap: dist >= r_i + r_j

    def boundary_constraints(vars_vec):
        constraints = []
        for i in range(n):
            x = vars_vec[3 * i]
            y = vars_vec[3 * i + 1]
            r = vars_vec[3 * i + 2]
            # x - r >= 0
            constraints.append(x - r)
            # 1 - (x + r) >= 0
            constraints.append(1 - x - r)
            # y - r >= 0
            constraints.append(y - r)
            # 1 - (y + r) >= 0
            constraints.append(1 - y - r)
        return np.array(constraints)

    def overlap_constraints(vars_vec):
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                x_i, y_i, r_i = vars_vec[3 * i:3 * i + 3]
                x_j, y_j, r_j = vars_vec[3 * j:3 * j + 3]
                
                dx = x_i - x_j
                dy = y_i - y_j
                dist_sq = dx**2 + dy**2
                min_dist = r_i + r_j
                
                # We want dist >= min_dist <=> dist^2 >= min_dist^2
                # Constraint: dist^2 - min_dist^2 >= 0
                constraints.append(dist_sq - min_dist**2)
        return np.array(constraints)

    # Combine constraints
    def all_constraints(vars_vec):
        return np.concatenate([boundary_constraints(vars_vec), overlap_constraints(vars_vec)])

    # Define bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Create constraint object for scipy
    # NonlinearConstraint requires fun(x) >= lb and fun(x) <= ub
    # We want constraints >= 0
    nonlin_constraint = NonlinearConstraint(all_constraints, 0, np.inf)

    # Run optimization
    # Trust-constr is good for bounded nonlinear problems
    res = minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=[nonlin_constraint], 
                   options={'verbose': 1, 'maxiter': 1000})

    # Extract solution
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = res.x[3 * i]
        final_centers[i, 1] = res.x[3 * i + 1]
        final_radii[i] = res.x[3 * i + 2]

    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
