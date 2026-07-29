# sol_000109 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 81b841bb) state=6d54373d sum of radii=2.503409 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

# Global constants for the problem
N_CIRCLES = 26
NUM_VARS = 2 * N_CIRCLES + 1  # 52 coordinates + 1 radius

def objective_function(v):
    """
    Objective to minimize: -radius (to maximize radius).
    v is a vector of shape (53,).
    v[0:52] are x,y coordinates.
    v[52] is the radius r.
    """
    return -v[NUM_VARS - 1]

def constraint_function(v):
    """
    Returns an array of inequality constraint values.
    All values must be >= 0 for the solution to be feasible.
    Constraints:
    1. Boundary: x_i >= r, 1-x_i >= r, y_i >= r, 1-y_i >= r
    2. Non-overlap: dist(i,j) >= 2r  =>  dist^2 >= 4r^2
    """
    r = v[NUM_VARS - 1]
    coords = v[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    
    constraints_list = []
    
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Boundary constraints
    # x - r >= 0
    constraints_list.extend(x - r)
    # 1 - x - r >= 0
    constraints_list.extend(1.0 - x - r)
    # y - r >= 0
    constraints_list.extend(y - r)
    # 1 - y - r >= 0
    constraints_list.extend(1.0 - y - r)
    
    # Pairwise distance constraints
    # We need (x_i - x_j)^2 + (y_i - y_j)^2 - 4r^2 >= 0 for all i < j
    # Compute squared Euclidean distances between all pairs
    # diff shape: (N, N, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Extract upper triangle elements (i < j)
    # triu_indices returns (rows, cols)
    rows, cols = np.triu_indices(N_CIRCLES, k=1)
    pair_dist_sq = dist_sq[rows, cols]
    
    # Constraint: dist_sq - 4r^2 >= 0
    constraints_list.extend(pair_dist_sq - 4.0 * r**2)
    
    return np.array(constraints_list)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    
    # 1. Compute a valid fallback solution (Grid packing)
    # A 6x5 grid fits 30 circles with radius 1/12.
    # We take 26 circles.
    r_fb = 1.0 / 12.0
    centers_fb = np.zeros((N_CIRCLES, 2))
    count = 0
    
    # x coordinates: r, 3r, 5r, 7r, 9r (5 points) -> Spacing 2r
    # y coordinates: r, 3r, ..., 11r (6 points) -> Spacing 2r
    x_coords = [r_fb + 2.0 * r_fb * k for k in range(5)]
    y_coords = [r_fb + 2.0 * r_fb * k for k in range(6)]
    
    for y in y_coords:
        for x in x_coords:
            centers_fb[count] = [x, y]
            count += 1
            if count == N_CIRCLES:
                break
        if count == N_CIRCLES:
            break
    
    radii_fb = np.full(N_CIRCLES, r_fb)
    best_r = r_fb
    best_centers = centers_fb
    best_radii = radii_fb
    
    # 2. Optimization Setup
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)]
    constraints = {
        'type': 'ineq',
        'fun': constraint_function
    }
    
    # 3. Run optimization with multiple restarts
    num_restarts = 15
    
    for seed in range(num_restarts):
        np.random.seed(seed * 1000 + 42)
        
        # Initialize centers randomly
        init_centers = np.random.rand(N_CIRCLES, 2)
        # Initialize radius with a reasonable guess
        init_r = 0.05 
        
        initial_vars = np.concatenate([init_centers.flatten(), [init_r]])
        
        try:
            result = opt.minimize(
                objective_function,
                initial_vars,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False}
            )
            
            # Check if optimization found a better radius
            current_r = result.x[NUM_VARS - 1]
            
            if current_r > best_r:
                best_r = current_r
                best_centers = result.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
                best_radii = np.full(N_CIRCLES, best_r)
                
        except Exception:
            continue
            
    return best_centers, best_radii, N_CIRCLES * best_r
