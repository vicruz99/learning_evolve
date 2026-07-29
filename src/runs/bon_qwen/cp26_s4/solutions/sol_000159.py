# sol_000159 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2296af5d) state=25b4ad85 sum of radii=0.003250 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses scipy.optimize.minimize with a penalty method on a hexagonal initialization.
    """
    n = 26
    
    # Objective function: minimize -sum(r) + penalty
    # params is a 1D array of size 78: [x0, y0, x1, y1, ..., x25, y25, r0, r1, ..., r25]
    def objective(params):
        # Unflatten
        c = params[:2*n].reshape((n, 2))
        r = params[2*n:]
        
        # Penalty for boundary violations
        # Constraints: r <= x <= 1-r  =>  x-r >= 0  AND  1-r-x >= 0
        # We penalize max(0, violation)^2
        pen = 0.0
        
        # Left boundary: x >= r  =>  r - x <= 0
        pen += np.sum(np.maximum(r - c[:, 0], 0)**2)
        # Right boundary: x <= 1 - r  =>  x + r - 1 <= 0
        pen += np.sum(np.maximum(c[:, 0] + r - 1.0, 0)**2)
        # Bottom boundary: y >= r
        pen += np.sum(np.maximum(r - c[:, 1], 0)**2)
        # Top boundary: y <= 1 - r
        pen += np.sum(np.maximum(c[:, 1] + r - 1.0, 0)**2)
        
        # Penalty for overlaps
        # Constraint: dist(i, j) >= r_i + r_j  =>  dist - (r_i + r_j) >= 0
        # Compute pairwise distances
        # diff shape: (n, n, 2)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # r_sum shape: (n, n)
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        overlap = r_sum - dists
        # Only positive overlaps (violations) are penalized
        pen += np.sum(np.maximum(overlap, 0)**2)
        
        # Objective value: we want to maximize sum(r), so minimize -sum(r)
        # Penalty weight lambda
        lam = 1000.0
        return -np.sum(r) + lam * pen

    # Initialization: Hexagonal Lattice
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05) # Start with a safe small radius
    
    # Generate points in a hexagonal pattern
    points = []
    r_start = 0.05
    y = r_start
    row = 0
    # Try to fill the square
    while len(points) < n:
        # Shift every other row
        x = r_start if row % 2 == 0 else 2 * r_start
        while x <= 1.0 - r_start:
            points.append([x, y])
            x += 2 * r_start
        y += math.sqrt(3) * r_start
        row += 1
    
    # Fallback if grid generation didn't yield enough points (unlikely for n=26)
    if len(points) < n:
        points = np.random.rand(n, 2).tolist()
    else:
        points = points[:n]
        
    centers = np.array(points)
    initial_params = np.hstack([centers.flatten(), radii])
    
    best_sum_radii = -1.0
    best_c = None
    best_r = None
    
    try:
        from scipy.optimize import minimize
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
        
        # Run multiple restarts to escape local minima
        num_restarts = 10
        for i in range(num_restarts):
            # Create a perturbed copy of initial params
            current_params = initial_params.copy()
            
            # Add noise to positions
            noise = np.random.uniform(-0.05, 0.05, 2*n)
            current_params[:2*n] += noise
            
            # Slightly vary radii
            current_params[2*n:] = 0.05 + np.random.uniform(-0.01, 0.01, n)
            current_params[2*n:] = np.clip(current_params[2*n:], 0, 0.5)
            
            res = minimize(objective, current_params, method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-12})
            
            # Extract result
            c_temp = res.x[:2*n].reshape((n, 2))
            r_temp = res.x[2*n:]
            s = np.sum(r_temp)
            
            # Keep the best result found
            if s > best_sum_radii:
                best_sum_radii = s
                best_c = c_temp.copy()
                best_r = r_temp.copy()
                
    except ImportError:
        # Fallback if scipy is not available (though allowed)
        # Use the initial grid configuration
        best_c = centers
        best_r = radii

    # Final Validation and Adjustment
    # The optimizer minimizes penalty, but we need strict non-overlap for validation.
    # We calculate the maximum violation and shrink radii uniformly.
    
    if best_c is not None:
        c = best_c
        r = best_r
        
        # Calculate max pairwise overlap
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf) # Ignore self-distance
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        overlap = r_sum - dists
        max_overlap = np.max(overlap)
        
        # Calculate max boundary violation
        b_ov = np.maximum(r - c[:, 0], 0)
        b_ov = np.maximum(b_ov, np.maximum(r - c[:, 1], 0))
        b_ov = np.maximum(b_ov, np.maximum(c[:, 0] + r - 1.0, 0))
        b_ov = np.maximum(b_ov, np.maximum(c[:, 1] + r - 1.0, 0))
        max_b_ov = np.max(b_ov)
        
        # Margin to ensure strict validity (plus epsilon for float precision)
        margin = max(max_overlap, max_b_ov) + 1e-12
        
        # Shrink radii
        r_final = np.maximum(r - margin, 0)
        
        # Ensure centers are within valid range [r, 1-r] for the new radii
        # Note: If r shrinks, the valid range [r, 1-r] shrinks.
        # If centers were valid for old r, they might be out for new r if we didn't adjust.
        # However, if r_new < r_old, then [r_new, 1-r_new] contains [r_old, 1-r_old].
        # So if c was in [r_old, 1-r_old], it is definitely in [r_new, 1-r_new].
        # We clip just to be safe against numerical noise.
        c[:, 0] = np.clip(c[:, 0], r_final, 1 - r_final)
        c[:, 1] = np.clip(c[:, 1], r_final, 1 - r_final)
        
        return c, r_final, np.sum(r_final)
    else:
        return centers, radii, 0.0
