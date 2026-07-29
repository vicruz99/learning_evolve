# sol_000165 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3abb93ac) state=7e203b63 sum of radii=2.608631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def boundary_constraints(p):
    """
    Computes boundary constraints for all circles.
    Returns a vector of constraint values. 
    For validity, all values must be >= 0.
    Constraints: x-r >= 0, 1-(x+r) >= 0, y-r >= 0, 1-(y+r) >= 0
    """
    n = len(p) // 3
    r = p[:n]
    centers = p[n:].reshape((n, 2))
    x = centers[:, 0]
    y = centers[:, 1]
    
    c1 = x - r
    c2 = 1 - (x + r)
    c3 = y - r
    c4 = 1 - (y + r)
    
    return np.concatenate([c1, c2, c3, c4])

def overlap_constraints(p):
    """
    Computes non-overlap constraints for all pairs of circles.
    Returns a vector of constraint values.
    Constraint: dist^2 - (r_i + r_j)^2 >= 0
    """
    n = len(p) // 3
    r = p[:n]
    centers = p[n:].reshape((n, 2))
    
    # Compute pairwise squared distances
    # diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Compute (r_i + r_j)^2
    sum_r = r[:, np.newaxis] + r[np.newaxis, :]
    min_dist_sq = sum_r**2
    
    # We only need the upper triangle (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    constraints = dist_sq[mask] - min_dist_sq[mask]
    
    return constraints

def all_constraints(p):
    """
    Combines boundary and overlap constraints.
    """
    c_bound = boundary_constraints(p)
    c_overlap = overlap_constraints(p)
    return np.concatenate([c_bound, c_overlap])

def objective_func(p):
    """
    Objective function: Maximize sum of radii.
    Minimize negative sum.
    """
    n = len(p) // 3
    return -np.sum(p[:n])

def run_packing():
    n = 26
    
    # Initial guess: Hexagonal packing configuration
    # 6 rows with counts [5, 4, 5, 4, 5, 3] sum to 26.
    # This provides a dense initial layout.
    counts = [5, 4, 5, 4, 5, 3]
    r_est = 0.092  # Estimated radius that fits in the square
    
    centers = []
    y = r_est
    for i, count in enumerate(counts):
        # Shift every other row by r to form hexagonal lattice
        shift = r_est if i % 2 == 1 else 0
        x_start = r_est + shift
        for j in range(count):
            x = x_start + j * 2 * r_est
            centers.append([x, y])
        y += r_est * np.sqrt(3)
    
    centers = np.array(centers)
    radii = np.full(n, r_est)
    
    # Variables: [r_0, ..., r_25, x_0, y_0, ..., x_25, y_25]
    x0 = np.concatenate([radii, centers.flatten()])
    
    # Bounds: radii in [0, 0.5], coordinates in [0, 1]
    bounds = [(0, 0.5)] * n + [(0, 1)] * (2 * n)
    
    # Define constraints
    cons = {'type': 'ineq', 'fun': all_constraints}
    
    # Run optimization using SLSQP
    res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False})
    
    if res.success:
        r_opt = res.x[:n]
        c_opt = res.x[n:].reshape((n, 2))
    else:
        # Fallback to initial guess if optimization fails
        r_opt = radii
        c_opt = centers
        
    sum_r = np.sum(r_opt)
    return c_opt, r_opt, sum_r
