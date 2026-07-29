# sol_000251 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5bb01f44) state=a18b56aa sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(vars_flat):
    """
    Objective function to minimize.
    We want to maximize sum of radii, so we minimize negative sum.
    """
    n = 26
    # vars_flat structure: [x1, y1, ..., x26, y26, r1, ..., r26]
    radii = vars_flat[2*n:]
    return -np.sum(radii)

def constraint_func(vars_flat):
    """
    Constraint function.
    Returns an array of values that must be >= 0.
    """
    n = 26
    centers = vars_flat[:2*n].reshape(n, 2)
    radii = vars_flat[2*n:]
    
    # Boundary constraints
    # x >= r  => x - r >= 0
    c1 = centers[:, 0] - radii
    # x <= 1-r => 1 - x - r >= 0
    c2 = 1.0 - centers[:, 0] - radii
    # y >= r => y - r >= 0
    c3 = centers[:, 1] - radii
    # y <= 1-r => 1 - y - r >= 0
    c4 = 1.0 - centers[:, 1] - radii
    boundary = np.concatenate([c1, c2, c3, c4])
    
    # Distance constraints
    # dist^2 >= (r_i + r_j)^2
    # Compute squared distance matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Compute squared sum of radii matrix
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    # We only need constraints for i < j (upper triangle)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dist_constraints = (dist_sq - r_sum_sq)[mask]
    
    return np.concatenate([boundary, dist_constraints])

def run_packing():
    n = 26
    
    # Initialization: Hexagonal grid generation
    # We use a reference radius r_ref = 0.09 to place points.
    # This spacing allows circles to touch but we will start with slightly smaller radii.
    r_ref = 0.09
    dy = r_ref * np.sqrt(3)
    dx = 2 * r_ref
    
    points = []
    j = 0
    while True:
        y = r_ref + j * dy
        if y > 1 - r_ref:
            break
        shift = r_ref if (j % 2 == 1) else 0.0
        i = 0
        while True:
            x = r_ref + i * dx + shift
            if x > 1 - r_ref:
                break
            points.append([x, y])
            i += 1
        j += 1
        
    # We expect around 30 points for r_ref=0.09. Select the first 26.
    init_centers = np.array(points[:26])
    
    # Initial radii slightly smaller than r_ref to ensure strict feasibility
    # for the optimizer start (slack in constraints).
    init_radii = np.full(n, r_ref * 0.95)
    
    # Flatten initial guess
    x0 = np.concatenate([init_centers.flatten(), init_radii])
    
    # Bounds for variables
    # Centers x, y in [0, 1]
    # Radii r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # Constraints definition
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Run optimization to maximize sum of radii (minimize negative sum)
    # SLSQP is a good choice for constrained non-linear optimization.
    res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False})
    
    # Extract results
    final_centers = res.x[:2*n].reshape(n, 2)
    final_radii = res.x[2*n:]
    
    # Ensure radii are non-negative (safety check)
    final_radii = np.maximum(final_radii, 0.0)
    
    sum_radii = float(np.sum(final_radii))
    
    return final_centers, final_radii, sum_radii
