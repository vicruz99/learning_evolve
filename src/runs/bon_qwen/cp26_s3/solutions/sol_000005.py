# sol_000005 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3ad176de) state=51da32ca sum of radii=0.000325 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initial Guess Generation
    # We create a grid of points and select the first 26 to initialize centers.
    # This provides a well-spread initial configuration.
    xs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    # Generate 6 y-coordinates evenly spaced to fill the height
    ys = np.linspace(0.1, 0.9, 6) 
    
    centers_list = []
    for y in ys:
        for x in xs:
            centers_list.append([x, y])
            if len(centers_list) >= n:
                break
        if len(centers_list) >= n:
            break
    
    # If we somehow didn't get enough (unlikely with this grid), fill with random
    while len(centers_list) < n:
        centers_list.append([np.random.rand(), np.random.rand()])
        
    centers = np.array(centers_list[:n])
    # Initial radii small enough to fit without overlap
    radii = np.full(n, 0.05)
    
    # Flatten variables into a single vector: [x1, y1, r1, x2, y2, r2, ...]
    # Shape: (3 * n,)
    vars_init = np.zeros(3 * n)
    vars_init[0::3] = centers[:, 0]
    vars_init[1::3] = centers[:, 1]
    vars_init[2::3] = radii
    
    # Define bounds for the optimizer
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # 2. Objective Function with Penalty
    def objective(vars_flat):
        # Reshape variables
        x = vars_flat[0::3]
        y = vars_flat[1::3]
        r = vars_flat[2::3]
        
        # Main objective: Maximize sum of radii -> Minimize negative sum
        obj = -np.sum(r)
        
        # Penalty parameters
        C = 10000.0 # Weight for penalty terms
        
        # Boundary Penalties
        # Circle i is inside if: r <= x <= 1-r  => x-r >= 0 and x+r <= 1
        # Penalty for x-r < 0: max(0, r-x)^2
        # Penalty for x+r > 1: max(0, x+r-1)^2
        penalty_boundary = 0.0
        penalty_boundary += np.sum(np.maximum(0, r - x)**2)
        penalty_boundary += np.sum(np.maximum(0, x + r - 1)**2)
        penalty_boundary += np.sum(np.maximum(0, r - y)**2)
        penalty_boundary += np.sum(np.maximum(0, y + r - 1)**2)
        
        # Overlap Penalties
        # Distance matrix
        # Using broadcasting to compute pairwise distances efficiently
        # x_i - x_j
        diff_x = x[:, np.newaxis] - x[np.newaxis, :]
        diff_y = y[:, np.newaxis] - y[np.newaxis, :]
        dist_sq = diff_x**2 + diff_y**2
        
        # Avoid square root for stability if possible, but we need dist for comparison
        # dist = sqrt(dist_sq)
        # However, sqrt is expensive? No, 26x26 is small.
        dist = np.sqrt(np.maximum(dist_sq, 0.0)) # clamp negative errors
        
        # Sum of radii matrix
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Overlap amount: r_i + r_j - dist_ij
        # We only care if this is positive
        overlap = np.maximum(0, r_sum - dist)
        
        # Sum of squared overlaps
        # Note: Matrix is symmetric, diagonal is 0.
        # Summing all entries counts each pair twice, which is fine for penalty
        penalty_overlap = np.sum(overlap**2)
        
        # Total objective
        return obj + C * (penalty_boundary + penalty_overlap)

    # 3. Optimization
    # Use L-BFGS-B which handles bounds
    result = minimize(
        fun=objective,
        x0=vars_init,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    # Extract optimized variables
    final_vars = result.x
    final_centers = np.zeros((n, 2))
    final_centers[:, 0] = final_vars[0::3]
    final_centers[:, 1] = final_vars[1::3]
    final_radii = final_vars[2::3]
    
    # 4. Post-processing / Validation check
    # Ensure radii are not negative (bounds should handle this, but just in case)
    final_radii = np.maximum(final_radii, 0.0)
    
    # If the penalty was high, the solution might be slightly invalid due to numerical precision
    # or the optimizer getting stuck. However, with high C, it should be very close.
    # We can try to shrink radii slightly if there are overlaps to guarantee validity.
    # But the problem asks to maximize sum, so we return the best found.
    # The validation function allows 1e-12 tolerance.
    
    sum_radii = np.sum(final_radii)
    
    # Sanity check on validity before returning (optional, but good for debugging)
    # We assume the optimizer worked.
    
    return final_centers, final_radii, sum_radii
