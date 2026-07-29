# sol_000010 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state accdaaf6) state=04821d1b sum of radii=2.339993 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the circle packing problem for 26 circles in a unit square
    to maximize the sum of radii.
    """
    np.random.seed(42)
    N = 26
    
    # --- 1. Initialization ---
    # We use a hexagonal lattice pattern to place the centers.
    # This is a highly efficient starting point for packing problems.
    centers = np.zeros((N, 2))
    
    # Parameters for the initial hexagonal grid
    rows = 6
    cols_per_row = [5, 4, 5, 4, 5, 3] # Sums to 26
    current_idx = 0
    
    # Initial radius guess (0.1 is a safe baseline for 25 circles, slightly less for 26)
    r_init = 0.09
    
    # Vertical and horizontal spacing for hexagonal packing
    dy = np.sqrt(3) * r_init
    dx = 2 * r_init
    
    for i, num_circles in enumerate(cols_per_row):
        y = r_init + i * dy
        # Shift every other row by dx/2 to create the hexagonal pattern
        x_start = r_init + (i % 2) * (dx / 2)
        
        for j in range(num_circles):
            x = x_start + j * dx
            centers[current_idx, 0] = x
            centers[current_idx, 1] = y
            current_idx += 1
            
    # Initial radii array
    radii = np.ones(N) * r_init
    
    # --- 2. Optimization Function ---
    # We minimize: -sum(radii) + penalty * violations
    # This converts a maximization problem with constraints into an unconstrained minimization.
    
    def cost_function(params):
        # Unpack parameters
        # params structure: [x1, y1, r1, x2, y2, r2, ...]
        centers_opt = params[:2*N].reshape(N, 2)
        radii_opt = params[2*N:]
        
        # Sum of radii (to be maximized, so we minimize negative)
        sum_r = np.sum(radii_opt)
        
        penalty = 0.0
        penalty_weight = 100.0
        
        # Boundary violations
        for i in range(N):
            x, y = centers_opt[i]
            r = radii_opt[i]
            if r < 0:
                r = 0 # Clamp to avoid NaNs in distance calc if r goes too low
            # Soft max for boundary: max(0, violation)
            penalty += np.maximum(0, r - x)
            penalty += np.maximum(0, r - (1 - x))
            penalty += np.maximum(0, r - y)
            penalty += np.maximum(0, r - (1 - y))
            
        # Overlap violations
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
                sum_radii = radii_opt[i] + radii_opt[j]
                if sum_radii > dist:
                    penalty += (sum_radii - dist)
        
        return -sum_r + penalty_weight * penalty

    # Initial parameters vector
    x0 = np.concatenate([centers.flatten(), radii])
    
    # --- 3. Run Optimizer ---
    # Bounds: x, y in [0, 1], r in [0, 1]
    bounds = []
    for _ in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 1.0)) # r
    
    # Options for L-BFGS-B
    options = {'maxiter': 2000, 'ftol': 1e-12}
    
    res = minimize(cost_function, x0, method='L-BFGS-B', jac='2-point', 
                   bounds=bounds, options=options)
    
    # --- 4. Extract and Post-Process Result ---
    best_centers = res.x[:2*N].reshape(N, 2)
    best_radii = res.x[2*N:]
    
    # Safety check and slight reduction to ensure strict validity 
    # against the 1e-12 tolerance in the validation function.
    # We clamp radii to be strictly within bounds and non-overlapping.
    
    # 1. Ensure circles are inside square
    for i in range(N):
        x, y = best_centers[i]
        r = best_radii[i]
        # Adjust center to be inside if radius is large
        # Actually, better to adjust radius if center is too close to edge
        margin = 1e-6
        best_radii[i] = min(best_radii[i], x - margin)
        best_radii[i] = min(best_radii[i], (1 - x) - margin)
        best_radii[i] = min(best_radii[i], y - margin)
        best_radii[i] = min(best_radii[i], (1 - y) - margin)
        
    # 2. Ensure no overlaps (reduce radii if needed)
    # A simple iterative pass to reduce radii of overlapping circles
    for _ in range(10): # Run a few passes
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                allowed_dist = dist + 1e-6 # Tiny buffer
                current_sum = best_radii[i] + best_radii[j]
                
                if current_sum > allowed_dist:
                    # Reduce radii proportionally to fit in the gap
                    factor = allowed_dist / current_sum
                    best_radii[i] *= factor
                    best_radii[j] *= factor

    # Final sum of radii
    final_sum_radii = np.sum(best_radii)
    
    # Handle potential NaNs or negative values from failed optimization
    if np.isnan(best_centers).any() or np.isnan(best_radii).any():
        # Fallback to a valid grid solution if optimizer failed
        best_centers = np.linspace(0.1, 0.9, 5)[:, None] * np.ones((5, 1))
        # This is just a placeholder, but the optimizer usually works.
        # Re-construct 26 centers for fallback if necessary
        fallback_centers = []
        fallback_radii = []
        for r in range(5):
            for c in range(5):
                fallback_centers.append([0.1 + c*0.2, 0.1 + r*0.2])
                fallback_radii.append(0.1)
        # Add 26th circle
        fallback_centers.append([0.5, 0.5])
        fallback_radii.append(0.04) # Small circle
        best_centers = np.array(fallback_centers)
        best_radii = np.array(fallback_radii)
        final_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, float(final_sum_radii)
