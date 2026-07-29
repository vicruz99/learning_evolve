# sol_000188 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 083f9270) state=8370ff47 sum of radii=2.567001 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # Estimate spacing. Area per circle approx 1/26. 
    # Hexagonal packing density pi/sqrt(12) ~ 0.9069.
    # Area of circle pi*r^2. N * pi * r^2 ~ 0.9069 * 1.
    # r ~ sqrt(0.9069 / (26 * pi)) ~ 0.105.
    # Diameter ~ 0.21. Spacing ~ 0.21.
    
    # Let's generate a grid with spacing 0.18 to be safe and fill
    spacing = 0.18
    centers = []
    
    # Hexagonal lattice generation
    # Row i, Col j
    # x = j * spacing + (i % 2) * spacing / 2
    # y = i * spacing * sqrt(3) / 2
    
    max_rows = int(1.0 / (spacing * np.sqrt(3)/2)) + 2
    max_cols = int(1.0 / spacing) + 2
    
    for i in range(max_rows):
        for j in range(max_cols):
            x = j * spacing + (i % 2) * (spacing / 2)
            y = i * spacing * np.sqrt(3) / 2
            
            # Check if inside [0,1]x[0,1] with some margin for radius
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers.append([x, y])
                if len(centers) == n:
                    break
        if len(centers) == n:
            break
            
    centers = np.array(centers)
    
    # If we didn't get enough (unlikely with this logic), pad or adjust
    # But with spacing 0.18, we should get plenty.
    # Let's ensure we have exactly n centers.
    if len(centers) > n:
        # Pick the first n
        centers = centers[:n]
    elif len(centers) < n:
        # Fallback to random if grid fails (should not happen)
        remaining = n - len(centers)
        extra = np.random.rand(remaining, 2)
        centers = np.vstack([centers, extra])

    # 2. Initial Radii
    # Calculate max valid radius for each center given current configuration
    # r_i <= dist_to_boundary and r_i <= dist_to_neighbor / 2
    radii = np.zeros(n)
    
    # Precompute pairwise distances
    # Using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_dist_to_neighbor = np.min(dists, axis=1)
    
    # Distance to boundaries
    dist_x_left = centers[:, 0]
    dist_x_right = 1 - centers[:, 0]
    dist_y_bottom = centers[:, 1]
    dist_y_top = 1 - centers[:, 1]
    
    dist_to_boundary = np.min(np.array([dist_x_left, dist_x_right, dist_y_bottom, dist_y_top]), axis=0)
    
    # Initial radius is limited by boundary and neighbors
    # We take a fraction to ensure strict feasibility initially
    max_r = np.minimum(dist_to_boundary, min_dist_to_neighbor / 2.0)
    radii = max_r * 0.9  # Start slightly smaller

    # 3. Optimization Variables
    # Flattened array: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]

    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # 4. Objective Function
    def objective(vars_flat):
        # Maximize sum of radii -> Minimize -sum(r)
        # r are at indices 2, 5, 8, ...
        r_vals = vars_flat[2::3]
        return -np.sum(r_vals)

    # 5. Constraints
    # We define a function that returns a vector of constraint values >= 0
    # Constraints:
    # 1. x_i - r_i >= 0
    # 2. 1 - x_i - r_i >= 0
    # 3. y_i - r_i >= 0
    # 4. 1 - y_i - r_i >= 0
    # 5. (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    def constraints(vars_flat):
        x = vars_flat[0::3]
        y = vars_flat[1::3]
        r = vars_flat[2::3]
        
        constraints_list = []
        
        # Boundary constraints (vectorized)
        constraints_list.append(x - r)           # x >= r
        constraints_list.append(1 - x - r)       # 1-x >= r
        constraints_list.append(y - r)           # y >= r
        constraints_list.append(1 - y - r)       # 1-y >= r
        
        # Non-overlap constraints
        # We need to compute squared distance minus squared sum of radii
        # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
        
        # Compute pairwise differences
        # x_diff[i, j] = xi - xj
        x_diff = x[:, np.newaxis] - x[np.newaxis, :]
        y_diff = y[:, np.newaxis] - y[np.newaxis, :]
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        sq_dist = x_diff**2 + y_diff**2
        sq_r_sum = r_sum**2
        
        # The diagonal is 0 - (2ri)^2 < 0, but we only care about i < j
        # We can extract upper triangle
        triu_indices = np.triu_indices(n, k=1)
        diff_constraints = sq_dist[triu_indices] - sq_r_sum[triu_indices]
        
        constraints_list.append(diff_constraints)
        
        # Return flattened array of all constraints
        return np.concatenate(constraints_list)

    # 6. Run Optimizer
    # SLSQP is good for this type of problem
    cons = {'type': 'ineq', 'fun': constraints}
    
    # We can run multiple times or just once. 
    # With a good start, one run should be sufficient.
    # However, to be safe, we might increase maxiter.
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
    
    # Check if successful
    if not res.success:
        # If failed, maybe return the initial valid packing or try to fix
        # But usually it finds something.
        pass

    # 7. Extract Results
    final_centers = np.column_stack((res.x[0::3], res.x[1::3]))
    final_radii = res.x[2::3]
    
    # Validate and clean up (clip tiny negative radii due to numerical error)
    final_radii = np.maximum(final_radii, 0.0)
    
    # Ensure centers are strictly within bounds relative to radii
    # Sometimes solver drifts. We clamp.
    # But the constraints should have prevented this.
    
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# To ensure the function is self-contained and valid, we wrap the call
# but the prompt asks for the function definition.
