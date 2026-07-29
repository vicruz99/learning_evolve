# sol_000258 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a15173c5) state=dbe99660 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    """
    n_circles = 26
    
    # --- Helper Functions ---
    
    def objective(params):
        # params structure: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
        # We want to maximize sum of radii, so minimize negative sum.
        radii = params[2::3]
        return -np.sum(radii)

    def constraint_boundary(params):
        # Constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        # Returns an array of values, all must be >= 0
        constraints = []
        for i in range(n_circles):
            x = params[3*i]
            y = params[3*i+1]
            r = params[3*i+2]
            
            # x - r >= 0
            constraints.append(x - r)
            # 1 - (x + r) >= 0  => x + r <= 1
            constraints.append(1.0 - (x + r))
            # y - r >= 0
            constraints.append(y - r)
            # 1 - (y + r) >= 0 => y + r <= 1
            constraints.append(1.0 - (y + r))
        return np.array(constraints)

    def constraint_overlap(params):
        # Constraints: dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        constraints = []
        for i in range(n_circles):
            xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
            for j in range(i + 1, n_circles):
                xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                sum_r = ri + rj
                
                # We enforce dist >= sum_r + epsilon to be safe, or just dist >= sum_r
                # The validator uses 1e-12 tolerance.
                # Optimization constraint: dist_sq - sum_r^2 >= 0
                val = dist_sq - sum_r*sum_r
                constraints.append(val)
        return np.array(constraints)

    # --- Initialization ---
    
    # We try to initialize with a pattern that is somewhat dense.
    # A hexagonal-like packing is good.
    # We need 26 circles. 
    # Let's try to fit them in a grid that is slightly compressed or jittered.
    
    # Create a grid of points
    # 6 rows, 5 cols = 30 points. We pick 26.
    # Or just place them randomly but spread out.
    
    # Strategy: Place in a 5x5 grid (25) and one extra.
    # But 5x5 grid with r=0.1 is full. 
    # Start with small r so they can grow.
    
    np.random.seed(42) # For reproducibility
    
    centers = np.zeros((n_circles, 2))
    radii = np.ones(n_circles) * 0.04 # Initial small radius
    
    # Distribute centers in a hexagonal pattern roughly
    # Hexagonal spacing: dx = 2r, dy = sqrt(3)r. 
    # But we don't know final r. Let's just spread them out in the square.
    
    # Grid approach:
    # 6 rows, 5 columns.
    rows = 6
    cols = 5
    # We have 26 circles. 6*5 = 30. We can just pick 26 positions.
    # Or place 26 in a dense grid.
    
    idx = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if idx >= n_circles:
                break
            # Coordinates
            # Shift odd rows to make it hexagonal
            x = (c_idx + 0.5) / cols + (0.5 / cols if r_idx % 2 == 1 else 0)
            y = (r_idx + 0.5) / rows
            
            # Clamp to [0, 1]
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            centers[idx] = [x, y]
            radii[idx] = 0.04
            idx += 1
        if idx >= n_circles:
            break
            
    # Add some random jitter to avoid perfect grid symmetry which might be a local minimum
    # or to help the optimizer break out.
    jitter = np.random.uniform(-0.02, 0.02, size=centers.shape)
    centers += jitter
    centers = np.clip(centers, 0.05, 0.95)
    
    # Flatten params: [x0, y0, r0, x1, y1, r1, ...]
    params = np.zeros(3 * n_circles)
    for i in range(n_circles):
        params[3*i] = centers[i, 0]
        params[3*i+1] = centers[i, 1]
        params[3*i+2] = radii[i]

    # --- Optimization Setup ---
    
    # Bounds: x in [0, 1], y in [0, 1], r in [0, 1] (loose upper bound)
    # Actually r can be at most 0.5.
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    # Constraints
    cons = []
    
    # Boundary constraints
    cons.append({'type': 'ineq', 'fun': constraint_boundary})
    
    # Overlap constraints
    cons.append({'type': 'ineq', 'fun': constraint_overlap})
    
    # --- Run Optimization ---
    
    # SLSQP is a good choice for constrained non-linear optimization
    # We might need to run it a few times or with different seeds if it gets stuck.
    # But let's try one solid run with a good initial guess.
    
    result = opt.minimize(
        objective, 
        params, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons,
        options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
    )
    
    # Extract solution
    best_params = result.x
    best_centers = np.zeros((n_circles, 2))
    best_radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        best_centers[i, 0] = best_params[3*i]
        best_centers[i, 1] = best_params[3*i+1]
        best_radii[i] = best_params[3*i+2]
        
    # Ensure radii are non-negative (solver might drift slightly below 0 if bounds not tight enough, but we set bounds)
    best_radii = np.maximum(best_radii, 0.0)
    
    # Calculate sum of radii
    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, sum_radii
