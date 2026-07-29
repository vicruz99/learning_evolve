# sol_000013 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ad74c980) state=adc63e72 sum of radii=0.003263 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import scipy.optimize
import sys

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    """
    n_circles = 26
    
    # We will optimize variables: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    # Total 26 * 3 = 78 variables.
    
    def objective(params):
        # params is a flat array of shape (78,)
        # Reshape to (26, 3)
        pts = params.reshape(n_circles, 3)
        centers = pts[:, :2]
        radii = pts[:, 2]
        
        sum_radii = np.sum(radii)
        
        penalty = 0.0
        
        # Penalty 1: Boundary constraints
        # x - r >= 0  => x - r >= 0
        # x + r <= 1  => 1 - x - r >= 0
        # y - r >= 0
        # y + r <= 1
        
        # Violations
        left_viol = np.maximum(0, radii - centers[:, 0])
        right_viol = np.maximum(0, centers[:, 0] + radii - 1.0)
        bottom_viol = np.maximum(0, radii - centers[:, 1])
        top_viol = np.maximum(0, centers[:, 1] + radii - 1.0)
        
        boundary_pen = np.sum(left_viol**2 + right_viol**2 + bottom_viol**2 + top_viol**2)
        
        # Penalty 2: Overlap constraints
        # dist_ij >= r_i + r_j
        # dist_ij^2 >= (r_i + r_j)^2
        # We check violation: r_i + r_j - dist_ij > 0
        
        # Vectorized overlap check
        # Compute pairwise distances
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (26, 26, 2)
        # dist_sq = np.sum(diff**2, axis=2) # (26, 26)
        # This creates a large matrix, but for 26 it's fine.
        
        # To save memory/time, we can just compute upper triangle or iterate
        # But vectorized is faster in numpy.
        
        # Centers shape (26, 2)
        # Pairwise squared distances
        # Using broadcasting
        diff = centers[:, None, :] - centers[None, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Radii sum matrix
        rad_sum = radii[:, None] + radii[None, :]
        rad_sum_sq = rad_sum**2
        
        # Violation: rad_sum - sqrt(dist_sq) > 0
        # Or rad_sum^2 - dist_sq > 0 (if dist > 0)
        # Using rad_sum - dist is safer for gradients near 0?
        # But dist = sqrt(dist_sq). Gradient is 1/(2*dist).
        # rad_sum^2 - dist_sq is quadratic in radii, smooth.
        # But constraint is dist >= rad_sum <=> dist^2 >= rad_sum^2.
        # So violation is max(0, rad_sum^2 - dist_sq).
        # However, if dist is 0, this might be tricky, but circles won't be on top of each other.
        
        # Let's use the direct distance violation for better geometric interpretation
        dist = np.sqrt(dist_sq + 1e-12) # Add epsilon to avoid NaN grad
        overlap_viol = np.maximum(0, rad_sum - dist)
        
        # We only care about upper triangle, but summing all is just 2x.
        overlap_pen = np.sum(overlap_viol**2)
        
        # Penalty weights
        # We want to maximize sum_radii, so minimize -sum_radii
        # Total Loss = -sum_radii + w1 * boundary_pen + w2 * overlap_pen
        # We need weights high enough to force constraints.
        w1 = 1000.0
        w2 = 1000.0
        
        return -sum_radii + w1 * boundary_pen + w2 * overlap_pen

    def gradient(params):
        # Numerical gradient might be slow, L-BFGS-B computes approximate grad or we can supply.
        # For simplicity and robustness, let L-BFGS-B approximate or use finite diff.
        # But providing a custom grad is better.
        # However, writing gradient for max functions is complex.
        # We will rely on scipy's finite difference approximation or use a method that doesn't need grad.
        # 'Powell' or 'Nelder-Mead' don't need grad but are slow.
        # 'L-BFGS-B' is good with bounds.
        return None # Let scipy handle it

    # Bounds for variables:
    # x in [0, 1]
    # y in [0, 1]
    # r in [0, 1] (technically r <= 0.5, but 1 is safe upper bound)
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (radius cannot exceed 0.5)

    best_score = -np.inf
    best_params = None
    
    # Run multiple iterations with different initializations
    n_runs = 10
    
    for run in range(n_runs):
        # Initialize
        # Strategy: Grid with perturbation
        # Try to fit 26 circles.
        # A 5x5 grid has 25. Let's try to place 26 in a way that utilizes space.
        # Maybe a 5x5 grid plus one in the middle? 
        # But 5x5 grid radius 0.1 is tight.
        # Let's start with smaller radius and let optimizer grow them.
        
        # Create a grid of 26 points?
        # Sqrt(26) approx 5.1.
        # Maybe 6x5 grid?
        # Let's generate random positions in [0.1, 0.9] to avoid boundaries initially
        # and set radius to 0.05.
        
        rng = np.random.RandomState(42 + run)
        
        # Initial guess: Perturbed grid
        # Let's try to arrange them roughly evenly
        # 5 rows, roughly 5-6 cols
        cols = 6
        rows = 5 # 30 slots, we pick 26
        
        # Or just random
        x_init = rng.rand(n_circles) * 0.8 + 0.1 # [0.1, 0.9]
        y_init = rng.rand(n_circles) * 0.8 + 0.1
        r_init = np.full(n_circles, 0.05) # Start small
        
        # Better initialization: Hexagonal grid packing
        # Rows of circles.
        # Let's try to construct a valid packing manually for initialization.
        # 5 rows.
        # Row 1: 5 circles
        # Row 2: 5 circles (shifted)
        # Row 3: 5 circles
        # Row 4: 5 circles (shifted)
        # Row 5: 6 circles?
        # Total 26.
        
        # Let's try a specific layout that is likely close to optimal.
        # Maybe a distorted 5x5 grid + 1.
        
        # Let's stick to random but clustered?
        # Actually, a grid is safer.
        # Let's create a grid of 26 points.
        # 26 = 2 * 13. Maybe 2 rows of 13? No.
        # 26 is prime.
        # Maybe 5x5 grid (25) + 1.
        
        # Let's try to place them in a 5x5 grid pattern but with 26th circle.
        # 5x5 grid centers:
        # x = [0.1, 0.3, 0.5, 0.7, 0.9]
        # y = [0.1, 0.3, 0.5, 0.7, 0.9]
        # This gives 25 circles.
        # Add 1 circle at (0.5, 0.5)? Overlaps.
        # Add at center of a gap?
        # Gap center between (0.5, 0.5) neighbors?
        # Maybe (0.4, 0.4)?
        
        # Let's just use random initialization, it's robust enough with many runs.
        # But bias towards center?
        
        x_init = rng.rand(n_circles) * 0.8 + 0.1
        y_init = rng.rand(n_circles) * 0.8 + 0.1
        r_init = np.full(n_circles, 0.08) # Start with reasonable radius
        
        # Try to avoid initial overlap
        # Simple rejection sampling for initialization
        while True:
            valid = True
            # Check bounds
            if np.any(x_init < 0) or np.any(x_init > 1) or np.any(y_init < 0) or np.any(y_init > 1):
                valid = False
            if np.any(r_init < 0):
                valid = False
            
            # Check overlap (strictly, no overlap allowed for init to be good, but penalty handles it)
            # Let's just ensure they are not on top of each other
            for i in range(n_circles):
                for j in range(i+1, n_circles):
                    dist = math.hypot(x_init[i]-x_init[j], y_init[i]-y_init[j])
                    if dist < 0.01: # Too close
                        valid = False
                        break
                if not valid: break
            
            # Check boundary clearance
            for i in range(n_circles):
                if x_init[i] - r_init[i] < 0 or x_init[i] + r_init[i] > 1 or \
                   y_init[i] - r_init[i] < 0 or y_init[i] + r_init[i] > 1:
                    valid = False
                    break
            
            if valid:
                break
            
            # Regenerate
            x_init = rng.rand(n_circles) * 0.8 + 0.1
            y_init = rng.rand(n_circles) * 0.8 + 0.1
            # Reduce radius if hard to place
            r_init = np.full(n_circles, 0.05)

        params0 = np.concatenate([x_init, y_init, r_init]).flatten() # Wait, objective expects [x,y,r] structure
        # My objective reshapes to (26, 3). So params should be [x0, y0, r0, x1, y1, r1 ...]
        # But I generated separate arrays.
        # Let's interleave.
        
        init_params = []
        for i in range(n_circles):
            init_params.extend([x_init[i], y_init[i], r_init[i]])
        params0 = np.array(init_params)
        
        # Optimization
        # Using L-BFGS-B with bounds
        try:
            res = scipy.optimize.minimize(
                objective, 
                params0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            score = -res.fun # Objective was -sum_radii + penalty
            # But fun includes penalty.
            # We need to evaluate the actual sum of radii of the result.
            pts = res.x.reshape(n_circles, 3)
            current_sum_radii = np.sum(pts[:, 2])
            
            # Check validity roughly (penalty should be near 0)
            # We can re-evaluate objective components
            # But simply, if penalty was high, sum_radii might be inflated but invalid.
            # However, with high weights, optimizer should drive penalty to 0.
            
            # Let's store the result
            if current_sum_radii > best_score: # Actually we want max sum_radii
                # But we need to ensure validity.
                # The objective minimizes -sum + penalty.
                # If penalty is 0, then minimizing objective = maximizing sum.
                # If penalty > 0, the objective value is higher (worse).
                # But comparing -fun is not correct because of penalty.
                # We should compare the valid sum of radii.
                
                # Let's compute a "clean" score
                # If penalty is low, it's a valid candidate.
                penalty_val = res.fun + current_sum_radii
                if penalty_val < 1e-4: # Valid enough
                    if current_sum_radii > best_score:
                        best_score = current_sum_radii
                        best_params = res.x
                else:
                    # Maybe still update if it's better than current best valid?
                    # But invalid solutions are useless.
                    pass
                    
        except Exception as e:
            print(f"Optimization failed: {e}")
            continue

    if best_params is None:
        # Fallback: Return the last valid result or a simple grid
        # Generate a safe 5x5 grid + 1 small circle?
        # 5x5 grid r=0.09 fits.
        centers = np.zeros((26, 2))
        radii = np.full(26, 0.09)
        
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                idx += 1
        # 26th circle
        centers[25] = [0.5, 0.5] # Overlaps, reduce radius
        radii[25] = 0.01
        # Check overlaps for 26th?
        # Distance to (0.5, 0.5) from (0.5, 0.5) is 0.
        # From (0.3, 0.5) dist 0.2. r=0.09. 0.2 >= 0.09+0.01? Yes.
        # Actually (0.5, 0.5) is center of square.
        # Nearest centers in 5x5 grid (r=0.09): (0.5, 0.3), (0.5, 0.7), (0.3, 0.5), (0.7, 0.5).
        # Dist 0.2. Sum radii 0.09 + 0.01 = 0.10. 0.2 >= 0.1. OK.
        # But wait, 5x5 grid with r=0.09 has centers at 0.1, 0.3, 0.5, 0.7, 0.9.
        # Wait, 0.1 + 4*0.2 = 0.9.
        # (0.5, 0.5) is a center in the grid!
        # Ah, 5x5 grid includes (0.5, 0.5).
        # So 25 circles fill the grid.
        # We need a 26th.
        # Where to put it?
        # Maybe in a corner gap?
        # Or just shrink all.
        
        # Let's just return the optimized result if available, else a fallback.
        # But best_params is None means no valid solution found?
        # Let's try to run one more time with a specific grid init.
        
        # Grid init: 5x5 grid of radius 0.08, plus 26th circle?
        # Actually, 26 circles.
        # Maybe 6 rows?
        # Let's try a hexagonal init.
        
        # Re-run optimization with hexagonal init
        # This logic is inside the loop, but if best_params is None, we failed.
        # Let's ensure we have a solution.
        
        # Fallback solution:
        # 26 circles, radius 0.08.
        # Can we fit 26 circles of r=0.08?
        # Area ~ 26 * pi * 0.0064 ~ 0.52. Feasible.
        # Hexagonal packing density 0.9. 0.52 / 0.9 ~ 0.58 < 1. Yes.
        
        # Let's generate a valid hexagonal packing for fallback.
        # Rows: 5, 5, 5, 5, 5, 1? No.
        # 5, 5, 5, 5, 4, 2?
        # Let's just do a random search for a valid fallback.
        
        # Actually, the optimization loop should have found something.
        # If not, I'll construct one.
        
        # Construct a valid packing with r=0.08
        # 5 rows of 5 circles = 25.
        # 1 circle somewhere.
        # Shift rows to hexagonal.
        
        pass

    # If best_params is still None (unlikely if logic correct), force a valid return
    if best_params is None:
        # Simple grid solution with reduced radius
        centers = np.zeros((26, 2))
        radii = np.full(26, 0.08)
        # Place 25 in 5x5 grid
        for i in range(5):
            for j in range(5):
                centers[5*i + j] = [0.1 + i*0.2, 0.1 + j*0.2]
        # Place 26th in a gap?
        # Gap between (0.1, 0.1) and (0.3, 0.3)?
        # Maybe (0.2, 0.2)?
        # Dist to (0.1, 0.1) is sqrt(0.02) ~ 0.141.
        # r+r = 0.16. 0.141 < 0.16. Overlap.
        # Need smaller radius or better spot.
        # (0.2, 0.2) is too close.
        # Maybe (0.5, 0.5) is occupied.
        # Maybe put 26th circle at (0.5, 0.95)?
        # Dist to (0.5, 0.7) is 0.25. r+r = 0.16. OK.
        # Dist to (0.3, 0.9) is sqrt(0.2^2 + 0.05^2) = sqrt(0.0425) ~ 0.206. OK.
        # Dist to (0.7, 0.9) is same. OK.
        # Dist to (0.5, 0.7) is 0.25.
        # Top boundary: 0.95 + 0.08 = 1.03 > 1. Violation.
        # Reduce radius to 0.05 for 26th?
        
        # Let's just use the optimizer's best effort.
        # If it returns invalid, validation will fail.
        # But I'll try to make sure it's valid.
        
        # Re-init with a valid configuration
        centers = np.zeros((26, 2))
        radii = np.full(26, 0.085) # Try slightly larger
        
        # 5x5 grid
        for i in range(5):
            for j in range(5):
                centers[5*i + j] = [0.1 + i*0.2, 0.1 + j*0.2]
        
        # 26th circle
        # Try placing at (0.5, 0.5) - occupied.
        # Try placing in a "hole" of hexagonal packing?
        # The 5x5 grid is square packing.
        # Holes are at (0.2, 0.2), (0.2, 0.4)...
        # Distance to nearest grid points (0.1, 0.1), (0.3, 0.1), etc.
        # Dist = sqrt(0.1^2 + 0.1^2) = 0.1414.
        # Available space for radius r2: 0.1414 - r1 = 0.1414 - 0.085 = 0.0564.
        # So we can fit a circle of radius 0.056.
        # But we want to maximize sum.
        # Maybe shrink grid circles slightly to accommodate larger 26th?
        # Or just accept smaller 26th.
        
        # Let's optimize this specific config.
        # 25 circles at grid, 1 circle at (0.2, 0.2) with small radius.
        # Then run optimizer.
        
        centers[25] = [0.2, 0.2]
        radii[25] = 0.05
        
        # This is a valid starting point?
        # Check 25th circle (index 25) vs neighbors.
        # Nearest neighbors in grid: (0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3).
        # Dist to (0.1, 0.1) = 0.1414.
        # r_grid + r_26 = 0.085 + 0.05 = 0.135.
        # 0.1414 > 0.135. Valid.
        # Boundary: 0.2 - 0.05 = 0.15 > 0. Valid.
        
        # Now optimize this.
        params0 = []
        for i in range(26):
            params0.extend([centers[i, 0], centers[i, 1], radii[i]])
        params0 = np.array(params0)
        
        res = scipy.optimize.minimize(
            objective, 
            params0, 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        
        pts = res.x.reshape(26, 3)
        best_params = res.x
        best_score = np.sum(pts[:, 2])

    # Extract results
    final_pts = best_params.reshape(n_circles, 3)
    final_centers = final_pts[:, :2]
    final_radii = final_pts[:, 2]
    
    # Final validation and repair
    # If validation fails, try to shrink radii slightly
    # But we trust the optimizer with high penalty.
    
    # Just to be safe, clamp radii to be valid with a margin
    # But the optimizer should have handled it.
    
    return final_centers, final_radii, np.sum(final_radii)
