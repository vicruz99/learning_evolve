# sol_000093 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e7a6456) state=88d72bd1 sum of radii=2.513552 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    num_vars = 3 * n  # x, y, r for each circle

    # --- Objective Function ---
    def objective(x):
        radii = x[2*n:]
        return -np.sum(radii)

    # --- Constraints ---
    # We define a single function that returns an array of constraint values >= 0
    def constraints_func(x):
        centers = x[:2*n].reshape((n, 2))
        radii = x[2*n:]
        
        cons = []
        
        # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
        cons.append(centers[:, 0] - radii)      # x >= r
        cons.append(1.0 - centers[:, 0] - radii) # 1-x >= r
        cons.append(centers[:, 1] - radii)      # y >= r
        cons.append(1.0 - centers[:, 1] - radii) # 1-y >= r
        
        # Pairwise non-overlap constraints
        # dist^2 >= (r_i + r_j)^2  => dist^2 - (r_i + r_j)^2 >= 0
        # We compute this for all i < j
        # Vectorized approach for performance
        # Create index arrays for upper triangle
        i_idx, j_idx = np.triu_indices(n, k=1)
        
        dists_sq = np.sum((centers[i_idx] - centers[j_idx])**2, axis=1)
        radii_sums = radii[i_idx] + radii[j_idx]
        cons.append(dists_sq - radii_sums**2)
        
        return np.concatenate(cons)

    # --- Bounds ---
    # x, y in [0, 1], r in [0, 0.5] (since max radius in unit square is 0.5)
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    constraint_def = {'type': 'ineq', 'fun': constraints_func}

    # --- Initialization Strategies ---
    def get_initial_guess(method='hex', seed=0):
        np.random.seed(seed)
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.05) # Start with small valid radii

        if method == 'grid':
            # Square grid initialization
            # Try to fit in a grid, picking best spots
            # 6x5 grid = 30 spots, pick 26
            x_vals = np.linspace(0.15, 0.85, 5) # 5 cols
            y_vals = np.linspace(0.15, 0.85, 6) # 6 rows
            points = []
            for y in y_vals:
                for x in x_vals:
                    points.append([x, y])
            # Shuffle and pick first 26
            idx = np.random.choice(len(points), size=n, replace=False)
            centers = np.array(points)[idx]
            
        elif method == 'hex':
            # Hexagonal packing initialization
            # Row spacing sqrt(3)/2 * 2r approx 1.732 * 0.1 = 0.1732
            # Let's place centers with spacing 0.2 initially
            row_idx = 0
            col_idx = 0
            r_init = 0.12 # spacing approx 0.24
            x_start = 0.12
            y_start = 0.12
            x_spacing = 2 * r_init
            y_spacing = math.sqrt(3) * r_init
            
            # Fill rows
            row_count = 0
            filled = 0
            while filled < n:
                y = y_start + row_count * y_spacing
                # Offset every other row
                x_offset = (x_spacing / 2) * (row_count % 2)
                
                # Determine number of circles in this row
                # Max x is 0.88 approx
                max_cols = int((0.88 - x_start - x_offset) / x_spacing) + 1
                
                current_row_count = min(max_cols, n - filled)
                
                for i in range(current_row_count):
                    if filled < n:
                        x = x_start + x_offset + i * x_spacing
                        centers[filled] = [x, y]
                        filled += 1
                row_count += 1
                if y_spacing * row_count + y_start > 0.9:
                    break # Stop if y goes out of bounds roughly

        elif method == 'random':
            # Random valid initialization
            # Place points, ensure distance > 2*r_init
            r_init = 0.08
            min_dist = 2 * r_init
            centers = np.zeros((n, 2))
            placed = 0
            attempts = 0
            while placed < n and attempts < 10000:
                cx = np.random.uniform(r_init, 1 - r_init)
                cy = np.random.uniform(r_init, 1 - r_init)
                
                # Check distance to existing
                valid = True
                if placed > 0:
                    dists = np.linalg.norm(centers[:placed] - [cx, cy], axis=1)
                    if np.any(dists < min_dist + 1e-5): # Allow tiny overlap for placement? No, strict
                        valid = False
                
                if valid:
                    centers[placed] = [cx, cy]
                    placed += 1
                attempts += 1
            if placed < n:
                # Fallback to grid if random fails
                return get_initial_guess('grid', seed)

        # Construct full vector
        x0 = np.zeros(num_vars)
        x0[:2*n] = centers.flatten()
        x0[2*n:] = radii
        return x0

    # --- Optimization Loop ---
    best_x = None
    best_val = -np.inf
    
    # Try multiple starts
    methods = ['hex', 'grid', 'random', 'random', 'random', 'hex']
    
    # Specific hex seeds or variations could be useful, but random seeds in method handle it
    for i, method in enumerate(methods):
        try:
            x0 = get_initial_guess(method, seed=i)
            
            # Run optimizer
            # SLSQP is suitable for constrained non-linear optimization
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraint_def,
                options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success:
                val = -res.fun # Positive sum
                if val > best_val:
                    best_val = val
                    best_x = res.x
        except Exception as e:
            print(f"Optimization failed with {method}: {e}")
            continue

    # If no success, fallback to the first valid guess (though unlikely to be optimal)
    if best_x is None:
        best_x = get_initial_guess('grid', seed=0)

    # --- Post-processing and Validation ---
    centers = best_x[:2*n].reshape((n, 2))
    radii = best_x[2*n:]
    
    # Ensure radii are non-negative (optimizer bounds should handle this, but safety check)
    radii = np.maximum(radii, 0.0)
    
    # Recalculate sum
    sum_radii = np.sum(radii)
    
    # Just a sanity check for validity (optional, but good for debugging)
    # We assume the optimizer found a valid point satisfying constraints
    # However, numerical noise might cause tiny violations. 
    # The problem asks to return the result. The validation function is separate.
    
    return centers, radii, float(sum_radii)
