# sol_000066 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 16623584) state=b9e050c5 sum of radii=0.650000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- Step 1: Initialize centers in a Hexagonal Grid ---
    # A hexagonal grid provides a denser packing than a square grid.
    # We estimate the spacing based on fitting n circles.
    # Approximate density of hexagonal packing is pi/sqrt(12) ~ 0.9069
    # Area per circle ~ 1/n. Radius r ~ sqrt(1/(n * density * pi))
    # But for placement, we just generate points and scale.
    
    centers = []
    
    # Generate points in a staggered grid
    # Spacing dx = 1, dy = sqrt(3)/2 approx 0.866
    # We will scale these later
    row_y = 0
    dy = np.sqrt(3) / 2
    
    while len(centers) < n:
        row_x = 0
        if int(row_y / dy) % 2 == 1:
            row_x = 0.5 # Shift every other row
        
        while len(centers) < n:
            centers.append([row_x, row_y])
            row_x += 1
            # Check if we are getting too wide, though scaling handles it
            if row_x > 1.5: 
                break 
        row_y += dy
    
    centers = np.array(centers[:n])
    
    # Normalize centers to fit roughly inside [0,1]x[0,1]
    # Find bounding box
    min_x, min_y = np.min(centers, axis=0)
    max_x, max_y = np.max(centers, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale to fit within [0,1] with some padding
    scale_x = 1.0 / width
    scale_y = 1.0 / height
    scale = min(scale_x, scale_y) * 0.95 # 0.95 padding to allow optimizer room
    
    # Center the configuration
    centers = (centers - np.array([min_x, min_y])) * scale
    offset_x = (1.0 - max_x * scale) / 2.0
    offset_y = (1.0 - max_y * scale) / 2.0
    centers = centers + np.array([offset_x, offset_y])
    
    # --- Step 2: Define Objective and Constraints for Optimizer ---
    # We want to maximize the minimum separation distance 's' (which corresponds to 2*r).
    # Variables: flattened centers array of shape (n*2,).
    # We introduce a variable 's' (separation) to maximize.
    # Total variables: [x1, y1, ..., xn, yn, s] -> length 2n + 1
    
    def objective(vars_flat):
        # We want to maximize s, so minimize -s
        s = vars_flat[-1]
        return -s

    def constraint_separation(vars_flat):
        centers_opt = vars_flat[:2*n].reshape(n, 2)
        s = vars_flat[-1]
        
        # Check distance between all pairs
        # Vectorized distance calculation
        # dist^2 >= s^2
        # Using broadcasting for efficiency
        # O(n^2) which is fine for n=26
        diff = centers_opt[:, np.newaxis, :] - centers_opt[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # We need dist >= s for all i != j
        # diag is 0, ignore
        np.fill_diagonal(dist_sq, np.inf)
        min_dist_sq = np.min(dist_sq)
        
        return min_dist_sq - s**2

    def constraint_boundary(vars_flat):
        centers_opt = vars_flat[:2*n].reshape(n, 2)
        s = vars_flat[-1]
        
        # Centers must be at least s/2 from boundaries
        # x >= s/2  => x - s/2 >= 0
        # 1-x >= s/2 => 1 - x - s/2 >= 0
        # same for y
        
        # Return array of constraints
        # SLSQP handles inequality constraints g(x) >= 0
        
        # Constraint: x_i >= s/2
        c1 = centers_opt[:, 0] - s/2.0
        # Constraint: 1 - x_i >= s/2
        c2 = 1.0 - centers_opt[:, 0] - s/2.0
        # Constraint: y_i >= s/2
        c3 = centers_opt[:, 1] - s/2.0
        # Constraint: 1 - y_i >= s/2
        c4 = 1.0 - centers_opt[:, 1] - s/2.0
        
        return np.concatenate([c1, c2, c3, c4])

    # --- Step 3: Run Optimization ---
    # Initial guess
    initial_s = 0.05 # Small initial separation
    x0 = np.concatenate([centers.flatten(), [initial_s]])
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_separation},
        {'type': 'ineq', 'fun': constraint_boundary}
    ]
    
    # Bounds for centers [0, 1] and s > 0
    bounds = [(0, 1)] * (2*n) + [(0, 1)] # s bounded by 1 roughly
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-9})
    
    # --- Step 4: Extract Results ---
    if res.success:
        best_centers = res.x[:2*n].reshape(n, 2)
        best_s = res.x[-1]
        best_r = best_s / 2.0
    else:
        # Fallback to initial guess if optimization fails
        best_centers = centers
        best_s = initial_s
        best_r = initial_s / 2.0
        
    radii = np.full(n, best_r)
    sum_radii = np.sum(radii)
    
    return best_centers, radii, sum_radii
