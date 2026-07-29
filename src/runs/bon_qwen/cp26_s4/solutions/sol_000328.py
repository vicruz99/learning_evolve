# sol_000328 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 488bfafc) state=fe7d4f1b sum of radii=0.100701 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a staggered hexagonal lattice as an initial guess and refines it using numerical optimization.
    """
    
    # --- 1. Initial Hexagonal Placement ---
    # We distribute 26 circles into 5 staggered rows: 5, 6, 5, 6, 4.
    # This arrangement is geometrically dense for this count in a square.
    rows_count = [5, 6, 5, 6, 4]
    n_circles = sum(rows_count)
    
    # Estimate an initial radius (1.0 / 10 is a safe lower bound for 25 circles in a grid)
    r_initial = 0.10
    centers = []
    
    y_curr = r_initial  # Start at radius distance from bottom boundary
    
    # Vertical step height in hexagonal packing is r * sqrt(3)
    dy = r_initial * np.sqrt(3)
    
    for i, count in enumerate(rows_count):
        # Horizontal spacing in a row is 2*r. 
        # To center the row within the unit square [0, 1]:
        total_width = (count - 1) * 2 * r_initial
        x_start = (1.0 - total_width) / 2
        
        # Odd-indexed rows (1, 3...) are shifted horizontally by r to nestle in gaps
        if i % 2 == 1:
            x_start += r_initial
            
        for j in range(count):
            x = x_start + j * 2 * r_initial
            centers.append([x, y_curr])
        y_curr += dy

    centers = np.array(centers)

    # --- 2. Objective Function for Optimization ---
    # We maximize the sum of radii. To make this a smooth optimization problem, 
    # we use a "barrier" or "soft constraint" approach. 
    # We optimize variable X = [r, x1, y1, x2, y2, ..., x26, y26]
    
    n = n_circles
    
    def objective(X):
        r = X[0]
        c = X[1:].reshape(n, 2)
        
        # 1. Boundary Penalty: 
        # If a circle crosses a boundary, we subtract a large penalty.
        penalty = 0.0
        
        # Check boundaries [0, 1]
        # We use a squared penalty for smoothness
        for i in range(n):
            # Left/Right walls
            if c[i, 0] - r < 0:
                penalty += 1e6 * (c[i, 0] - r)**2
            if c[i, 0] + r > 1:
                penalty += 1e6 * (c[i, 0] + r - 1)**2
            
            # Bottom/Top walls
            if c[i, 1] - r < 0:
                penalty += 1e6 * (c[i, 1] - r)**2
            if c[i, 1] + r > 1:
                penalty += 1e6 * (c[i, 1] + r - 1)**2

        # 2. Overlap Penalty:
        # For every pair of circles, if distance < sum of radii, penalize.
        # Using a quadratic penalty for violations.
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((c[i] - c[j])**2))
                min_dist = 2 * r
                if dist < min_dist:
                    penalty += 1e6 * (min_dist - dist)**2
        
        # Objective: Maximize r (so we minimize -r) minus penalties
        # We want to maximize 26 * r, so we minimize -26 * r + penalty
        return -26 * r + penalty

    # --- 3. Optimization Execution ---
    # Initial state vector: [r, x1, y1, ..., x26, y26]
    x0 = np.concatenate([[r_initial], centers.flatten()])
    
    # Bounds: Radius must be positive. Coordinates can be anywhere, 
    # but penalties will push them into [0,1]. 
    # We set loose bounds to allow optimizer freedom.
    bounds = [(1e-9, 1.0)] + [(-0.1, 1.1) for _ in range(2 * n)]
    
    # Use L-BFGS-B for efficient bound-constrained optimization
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    
    # --- 4. Extract Results ---
    opt_r = result.x[0]
    opt_centers = result.x[1:].reshape(n, 2)
    
    # Final validation check (optional but good practice)
    # If the optimization failed to find a valid packing, the penalty would be high.
    # The objective value should be roughly -26 * r_opt.
    
    return (opt_centers, np.full(n, opt_r), 26 * opt_r)
