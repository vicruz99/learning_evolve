# sol_000023 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2175bd4f) state=06d7b407 sum of radii=2.280709 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # --- 1. Initialization: Hexagonal Grid ---
    # Start with a dense hexagonal packing to give the optimizer a strong head start.
    cols = 6
    rows = 5
    count = 0
    
    # Adjust parameters to fit 26 circles comfortably within [0,1]
    # Initial spacing estimation
    x_spacing = 0.16
    y_spacing = 0.16 * np.sqrt(3)/2 + 0.05
    
    for r in range(rows):
        for c in range(cols):
            if count >= n:
                break
            # Hexagonal offset
            x = 0.05 + c * x_spacing + (r % 2) * (x_spacing / 2)
            y = 0.05 + r * y_spacing
            
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers[count] = [x, y]
                # Assign a small initial radius
                radii[count] = 0.01
                count += 1
        if count >= n:
            break
            
    # Fallback if grid didn't fill (unlikely with these params)
    while count < n:
        centers[count] = [np.random.rand(), np.random.rand()]
        radii[count] = 0.01
        count += 1

    # --- 2. Optimization Loop ---
    # Iteratively optimize the position of each circle to maximize its radius,
    # treating other circles as fixed obstacles.
    
    num_iterations = 15
    
    # Cache radii for faster access in inner loop
    # (Though numpy array access is fast enough, this is cleaner logic)
    
    for _ in range(num_iterations):
        # Randomize order to prevent bias
        indices = np.random.permutation(n)
        
        for i in indices:
            # Pre-calculate neighbor data for this step to avoid recomputing in objective
            # Neighbors: centers and radii of all j != i
            neighbor_centers = np.delete(centers, i, axis=0)
            neighbor_radii = np.delete(radii, i)
            
            # Objective: Minimize negative max_radius
            def objective(pos):
                x, y = pos
                
                # 1. Check boundaries (distance to walls)
                # Max radius limited by x, 1-x, y, 1-y
                r_bound = min(x, 1.0 - x, y, 1.0 - y)
                
                # If out of bounds, return a large penalty
                if r_bound < -1e-6:
                    return 1000.0
                
                # 2. Check neighbors
                # Max radius limited by distance to neighbor center minus neighbor radius
                # r_i <= dist - r_j  =>  r_i <= sqrt((x-xj)^2 + (y-yj)^2) - rj
                
                # Vectorized distance calculation for speed
                dx = neighbor_centers[:, 0] - x
                dy = neighbor_centers[:, 1] - y
                dists = np.sqrt(dx*dx + dy*dy)
                
                # Available radius for each neighbor constraint
                r_neighbors = dists - neighbor_radii
                
                # The limiting factor is the minimum of all constraints
                r_max = min(r_bound, np.min(r_neighbors))
                
                # Radius cannot be negative
                if r_max < 0:
                    r_max = 0.0
                
                # We maximize r_max, so we minimize -r_max
                return -r_max

            # Initial guess for optimization
            x0 = centers[i].copy()
            
            # Bounds for x, y are [0, 1]
            bounds = [(0.0, 1.0), (0.0, 1.0)]
            
            # Use Nelder-Mead simplex method, good for non-smooth functions
            # and doesn't require gradient information.
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='Nelder-Mead', 
                    bounds=bounds,
                    options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7}
                )
                
                # Update center if a better position was found
                # We trust the solver's result if it converged or ran iterations
                if res.nit > 0:
                    centers[i] = res.x
                    
                    # Recompute the exact radius for circle i at the new position
                    # This ensures strict consistency with neighbors
                    x, y = centers[i]
                    r_val = min(x, 1.0 - x, y, 1.0 - y)
                    
                    # Check neighbors again to ensure non-overlap
                    # r_i + r_j <= dist  =>  r_i <= dist - r_j
                    for j in range(n):
                        if i == j: continue
                        dx = centers[j, 0] - x
                        dy = centers[j, 1] - y
                        dist = np.sqrt(dx*dx + dy*dy)
                        limit = dist - radii[j]
                        if limit < r_val:
                            r_val = limit
                    
                    radii[i] = max(0.0, r_val)
            except Exception:
                pass

    # --- 3. Final Validation and Correction ---
    # Sometimes numerical errors might leave tiny overlaps. 
    # We clamp radii to be strictly valid.
    for i in range(n):
        x, y = centers[i]
        # Boundary check
        r_bound = min(x, 1.0 - x, y, 1.0 - y)
        # Neighbor check
        for j in range(i + 1, n):
            dx = centers[j, 0] - x
            dy = centers[j, 1] - y
            dist = np.sqrt(dx*dx + dy*dy)
            # If dist < r_i + r_j, we have overlap.
            # We resolve by reducing radii proportionally or just ensuring valid sum.
            # Since we optimized sequentially, overlaps should be minimal/non-existent.
            # But to be safe for the validator:
            # We don't change positions here, just ensure radii are consistent.
            # Actually, the optimization ensures r_i <= dist - r_j.
            pass
        radii[i] = max(0.0, r_bound) # Boundary is always a hard constraint
        
        # Re-check neighbors to clamp if necessary (safety net)
        for j in range(n):
            if i == j: continue
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            max_r = dist - radii[j]
            if max_r < radii[i]:
                radii[i] = max_r

    sum_radii = np.sum(radii)
    
    # Final sanity check
    # If any radius is NaN or negative, fix it
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, sum_radii
