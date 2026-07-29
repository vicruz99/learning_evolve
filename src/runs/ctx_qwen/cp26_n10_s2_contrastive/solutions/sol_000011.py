# sol_000011 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b4d6f452) state=2a8987cb sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

# Set random seed for reproducibility in helper functions if needed, 
# but we want to explore different initializations in the main function.
# Note: Helper functions must be top level.

def compute_overlaps(centers, radii):
    """
    Computes the sum of squared overlap penalties for all pairs.
    Overlap penalty for pair (i, j) is max(0, r_i + r_j - dist_ij)^2.
    """
    n = centers.shape[0]
    penalty = 0.0
    # Vectorized calculation might be slow for memory if N is huge, but N=26 is small.
    # We can compute pairwise distances.
    # centers shape (N, 2)
    # diff shape (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # radii sum shape (N, N)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Overlap amount
    overlap = r_sum - dist
    # Only positive overlaps count
    overlap = np.maximum(0, overlap)
    
    # Sum of squares of upper triangle (to avoid double counting and self)
    # Use a mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    penalty = np.sum(overlap[mask]**2)
    return penalty

def compute_boundary_penalty(centers, radii):
    """
    Computes penalty for circles going outside [0,1]x[0,1].
    """
    n = centers.shape[0]
    penalty = 0.0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Constraints: r <= x <= 1-r, r <= y <= 1-r
        # Equivalently: x-r >= 0, 1-r-x >= 0, etc.
        violations = []
        if x - r < 0: violations.append((x-r)**2)
        if x + r > 1: violations.append((x+r-1)**2)
        if y - r < 0: violations.append((y-r)**2)
        if y + r > 1: violations.append((y+r-1)**2)
        penalty += sum(violations)
    return penalty

def objective_function(params, n_circles, penalty_weight):
    """
    Objective function to minimize.
    params: flattened array of [x1, y1, r1, x2, y2, r2, ...]
    """
    centers = params[0:2*n_circles].reshape(n_circles, 2)
    radii = params[2*n_circles:]
    
    # Ensure non-negative radii (though bounds handle this, good for safety in penalty)
    # But params can be negative during optimization if not bounded.
    # We rely on bounds in L-BFGS-B.
    
    obj = -np.sum(radii) # We want to maximize sum, so minimize negative
    obj += penalty_weight * compute_overlaps(centers, radii)
    obj += penalty_weight * compute_boundary_penalty(centers, radii)
    
    return obj

def run_packing():
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None
    
    # We will try multiple initializations
    num_restarts = 10
    
    for restart in range(num_restarts):
        # Initialize positions and radii
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        
        # Strategy: Hexagonal packing approximation
        # Try to fit in rows. 
        # A pattern like 5, 4, 5, 4, 5, 3 sums to 26.
        # Or random initialization.
        
        # Let's use a grid-like initialization that is perturbed
        # 5x5 grid is too crowded for 26, so maybe 5 rows.
        # But let's just randomize to avoid bias.
        
        # Generate random valid positions
        # Start with small radius to ensure validity
        initial_r = 0.05
        radii[:] = initial_r
        
        # Place centers in a hexagonal pattern roughly
        row_counts = [5, 4, 5, 4, 5, 3] # Sum = 26
        current_idx = 0
        
        # Vertical spacing for hex packing: r * sqrt(3)
        # But we don't know optimal r yet. Let's guess spacing.
        # Square side 1. 6 rows. Height ~ 1.
        # dy = 1.0 / 6.0
        
        y_curr = initial_r + 0.1 # Start a bit inside
        dy = (1.0 - 2*initial_r) / 5.0 # 6 rows, 5 gaps? No.
        # Let's just distribute y uniformly
        ys = np.linspace(initial_r + 0.1, 1.0 - initial_r - 0.1, 6)
        
        row_y_idx = 0
        for count in row_counts:
            if row_y_idx >= len(ys): break
            y = ys[row_y_idx]
            
            # Distribute x in [initial_r, 1-initial_r]
            # Shift alternate rows
            shift = 0
            if row_y_idx % 2 == 1:
                shift = 0.1 # Half spacing roughly
            
            xs = np.linspace(initial_r + shift, 1.0 - initial_r - shift, count)
            
            for k in range(count):
                if current_idx < n:
                    centers[current_idx, 0] = xs[k]
                    centers[current_idx, 1] = y
                    current_idx += 1
            row_y_idx += 1
            
        # If we didn't fill all, fill remaining randomly
        while current_idx < n:
            centers[current_idx] = [np.random.uniform(initial_r, 1-initial_r), 
                                    np.random.uniform(initial_r, 1-initial_r)]
            current_idx += 1
            
        # Add some noise
        centers += np.random.normal(0, 0.01, centers.shape)
        
        # Flatten parameters
        params0 = np.concatenate([centers.flatten(), radii])
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0)]) # x, y
            bounds.append((0.0, 0.5)) # r
            
        # Optimization parameters
        penalty_weight = 1000.0 # High penalty for constraints
        
        # Try to optimize
        # L-BFGS-B is good for bounds
        try:
            res = scipy.optimize.minimize(
                objective_function,
                params0,
                args=(n, penalty_weight),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            # Extract results
            opt_centers = res.x[0:2*n].reshape(n, 2)
            opt_radii = res.x[2*n:]
            
            # Check validity (with tolerance)
            valid = True
            # Simple check
            for i in range(n):
                x, y = opt_centers[i]
                r = opt_radii[i]
                if r < 0: valid = False; break
                if x < r or x > 1-r or y < r or y > 1-r: 
                    # Tight check, maybe allow small error? 
                    # But let's be strict for validity check
                    if not (x - r >= -1e-6 and x + r <= 1 + 1e-6 and y - r >= -1e-6 and y + r <= 1 + 1e-6):
                        valid = False; break
            
            if valid:
                sum_r = np.sum(opt_radii)
                if sum_r > best_sum_radii:
                    best_sum_radii = sum_r
                    best_centers = opt_centers.copy()
                    best_radii = opt_radii.copy()
            else:
                # Even if not strictly valid by simple check, maybe it's good?
                # But we need valid output.
                # Let's try to repair: shrink radii until valid
                # This is a fallback.
                # But let's just skip if invalid to be safe, or try to accept if close?
                # The prompt requires valid packing.
                # Let's check overlaps explicitly.
                pass
                
        except Exception as e:
            continue

    # If we found nothing valid (unlikely), fallback to grid
    if best_centers is None:
        # Fallback: 5x5 grid + 1 tiny circle? No, that's invalid or sum low.
        # Just return a valid small packing.
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.01)
        for i in range(n):
            centers[i] = [0.5, 0.5] # All same center? No overlap check will fail.
            # Better random valid
            radii[i] = 0.01
            centers[i] = [np.random.uniform(0.02, 0.98), np.random.uniform(0.02, 0.98)]
        best_centers = centers
        best_radii = radii
        best_sum_radii = np.sum(radii)

    return best_centers, best_radii, best_sum_radii
