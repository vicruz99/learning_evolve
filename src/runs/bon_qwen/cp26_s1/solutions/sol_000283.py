# sol_000283 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5952a474) state=060504ef sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Number of circles to pack
N_CIRCLES = 26

def get_initial_guess():
    """
    Generates a hexagonal grid initial guess for circle centers.
    This provides a good starting point close to a dense packing.
    """
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.full(N_CIRCLES, 0.1)
    
    # Parameters for hexagonal packing
    r_est = 0.09 
    y_curr = r_est
    count = 0
    row_idx = 0
    points = []
    
    # Generate points in a hexagonal lattice pattern
    while y_curr + r_est <= 1.0:
        # Shift odd rows horizontally by radius
        x_start = r_est + (row_idx % 2) * r_est
        x_curr = x_start
        
        while x_curr + r_est <= 1.0:
            points.append([x_curr, y_curr])
            x_curr += 2 * r_est
        
        y_curr += np.sqrt(3) * r_est
        row_idx += 1
        
    # If we don't have enough points, fill with random valid positions
    if len(points) < N_CIRCLES:
        for _ in range(N_CIRCLES - len(points)):
            points.append(np.random.uniform(0.05, 0.95, 2).tolist())
            
    # Take exactly N_CIRCLES points
    points = points[:N_CIRCLES]
    centers = np.array(points)
    
    return centers, radii

def objective_function(vars_vec):
    """
    Objective function to minimize.
    We want to maximize sum of radii, so we minimize negative sum.
    """
    # Radii are stored at indices 2, 5, 8, ... (3*i + 2)
    radii = vars_vec[2::3]
    return -np.sum(radii)

def boundary_constraints(vars_vec):
    """
    Returns array of boundary constraint values.
    Constraints must be >= 0 for 'ineq' type in SLSQP.
    """
    # Reshape vector to (N, 3) where columns are x, y, r
    c = vars_vec.reshape((N_CIRCLES, 3))
    x = c[:, 0]
    y = c[:, 1]
    r = c[:, 2]
    
    # 1. x - r >= 0 (Circle inside left boundary)
    cons = x - r
    # 2. 1 - x - r >= 0 (Circle inside right boundary)
    cons = np.concatenate((cons, 1.0 - x - r))
    # 3. y - r >= 0 (Circle inside bottom boundary)
    cons = np.concatenate((cons, y - r))
    # 4. 1 - y - r >= 0 (Circle inside top boundary)
    cons = np.concatenate((cons, 1.0 - y - r))
    
    return cons

def overlap_constraints(vars_vec):
    """
    Returns array of overlap constraint values.
    Constraints must be >= 0.
    For circles i and j: dist^2 - (r_i + r_j)^2 >= 0
    """
    c = vars_vec.reshape((N_CIRCLES, 3))
    centers = c[:, :2]
    radii = c[:, 2]
    
    # Compute pairwise squared distances
    # centers shape (N, 2)
    # diff shape (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Compute pairwise squared sum of radii
    # sum_r_sq shape (N, N)
    sum_r_sq = (radii[:, np.newaxis] + radii[np.newaxis, :])**2
    
    # Constraint value: dist_sq - sum_r_sq
    vals = dist_sq - sum_r_sq
    
    # Return only upper triangle (i < j) to avoid duplicates and self-interactions
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    return vals[mask]

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square.
    Returns (centers, radii, sum_radii).
    """
    # Define bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (upper bound 0.5 is safe for unit square)
    
    # Define constraints
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None
    
    # Run optimization multiple times with different initial perturbations
    # to escape local minima.
    np.random.seed(42)
    
    for run_idx in range(3):
        centers_init, radii_init = get_initial_guess()
        
        # Add small random perturbation to break symmetry
        centers_init += np.random.normal(0, 0.01, centers_init.shape)
        centers_init = np.clip(centers_init, 0.01, 0.99)
        
        # Flatten initial guess to optimization vector
        # Order: x0, y0, r0, x1, y1, r1, ...
        x0 = np.zeros(3 * N_CIRCLES)
        for i in range(N_CIRCLES):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = radii_init[i]
            
        try:
            # Use SLSQP method which handles non-linear constraints
            res = minimize(objective_function, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    res_reshaped = res.x.reshape((N_CIRCLES, 3))
                    best_centers = res_reshaped[:, :2].copy()
                    best_radii = res_reshaped[:, 2].copy()
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_centers is None:
        centers_init, radii_init = get_initial_guess()
        best_centers = centers_init
        best_radii = radii_init * 0.5 # Ensure valid small radii
        best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, best_sum_radii
