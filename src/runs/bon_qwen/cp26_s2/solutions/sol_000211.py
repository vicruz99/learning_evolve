# sol_000211 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8fb167b6) state=a5469e59 sum of radii=2.234091 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import random

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    
    # 1. Initialize centers in a hexagonal pattern
    # We try to fit 26 circles. A hexagonal pattern is efficient.
    # We estimate an initial radius r approx 0.1.
    # We will place points and then optimize.
    
    centers = np.zeros((n, 2))
    
    # Heuristic to place 26 points in a hexagonal grid
    # We iterate through possible row configurations or just fill a grid
    # and take the first 26 that fit, then optimize.
    
    # Let's try a dense hexagonal packing generation
    # Spacing dx = 2*r_est, dy = sqrt(3)*r_est
    # We don't know r_est exactly, but we can place points relative to each other
    # and scale them later or just place them in [0,1].
    
    # Let's use a random shuffle of a dense grid to break symmetry?
    # No, let's stick to hexagonal.
    
    # Try to fit rows. 
    # Row 0: y = r, x = r, 3r, ...
    # Row 1: y = r + sqrt(3)r, x = 2r, 4r, ...
    
    # Let's assume r=0.1 for placement
    r_est = 0.1
    dx = 2 * r_est
    dy = math.sqrt(3) * r_est
    
    points = []
    y = r_est
    row_idx = 0
    while len(points) < n:
        # Determine x start offset
        if row_idx % 2 == 0:
            x_start = r_est
        else:
            x_start = 2 * r_est # Shifted by r_est horizontally relative to previous row centers? 
            # Actually distance between (r, r) and (2r, r+dy) is sqrt(r^2 + 3r^2) = 2r. Correct.
        
        x = x_start
        while x <= 1 - r_est and len(points) < n:
            points.append([x, y])
            x += dx
        
        y += dy
        row_idx += 1
        
    # If we have more points, truncate
    if len(points) > n:
        points = points[:n]
    
    # If we have fewer (unlikely with this logic unless r is too big), we might have an issue, 
    # but r=0.1 is small enough.
    # Let's shuffle points slightly to avoid grid artifacts if optimization gets stuck?
    # Or just use them.
    centers = np.array(points)
    
    # 2. Optimization
    # We want to maximize sum of radii.
    # Let's use a simple iterative expansion and repulsion method.
    # It's robust and doesn't require complex derivative handling for inequalities.
    
    # Initialize radii to a small valid value
    # The minimum distance between any two points and to walls determines max possible equal radius.
    # But we start small.
    radii = np.full(n, 0.001)
    
    # Optimization loop
    # We will try to increase radii and move centers to resolve conflicts.
    
    # Parameters
    lr_move = 0.001  # Learning rate for moving centers
    lr_expand = 0.0005 # Learning rate for expanding radii
    max_iter = 10000
    temp = 1.0 # For simulated annealing acceptance (optional, maybe just greedy)
    
    # To make it faster and more effective, we can use a "force" approach.
    # Forces push centers apart if overlapping, and walls push centers in.
    # We also want to increase radii.
    
    # Let's implement a gradient-based local search with projection.
    # Actually, a simple randomized hill climbing with perturbation is safer to implement correctly.
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(best_radii)
    
    # Helper to check validity and calculate "pressure"
    # We want to maximize sum(r).
    # Constraint: dist(i, j) >= r_i + r_j
    # Constraint: r_i <= x_i <= 1-r_i, r_i <= y_i <= 1-r_i
    
    # Let's try to optimize radii first given fixed centers?
    # Given centers, max r_i is min(dist_to_wall, min_j (dist(i,j) - r_j)).
    # This is coupled.
    
    # Alternative: Optimize positions to maximize min_dist, then set radii?
    # No, we need unequal radii potentially.
    
    # Let's use a direct optimization of variables [x, y, r] using a penalty method or constraint handling.
    # Since scipy is available, let's use scipy.optimize.minimize.
    # We need to handle the inequality constraints.
    # Constraints are non-convex.
    
    # To make it work, we flatten variables.
    # vars = [x0, y0, r0, x1, y1, r1, ...]
    
    # However, with 26 circles, 78 vars and ~400 constraints might be slow.
    # Let's stick to a custom heuristic that is guaranteed to terminate and run fast.
    
    # Heuristic: "Expand and Relax"
    # 1. Increase all radii by small amount.
    # 2. If overlap, move circles apart.
    # 3. Repeat.
    
    # Better Heuristic:
    # Start with small radii.
    # Randomly pick a circle, try to increase radius.
    # If it overlaps, move it to a better spot (random walk until valid).
    # Or move it in direction of gradient of clearance.
    
    # Let's try a simpler approach:
    # Use the initial hexagonal placement.
    # Run a few iterations of "repulsive forces" to maximize minimum distance.
    # Then compute radii.
    # But radii don't have to be equal.
    
    # Let's try to optimize the sum of radii directly using a local search.
    
    current_centers = centers.copy()
    current_radii = np.full(n, 0.05) # Start with reasonable guess
    
    # Validate initial guess and fix if invalid
    # If current_radii are too big, shrink them.
    # Check validity
    def get_validity_score(c, r):
        score = np.sum(r)
        # Penalty for violations
        penalty = 0.0
        n = c.shape[0]
        for i in range(n):
            # Wall constraints
            if r[i] < 0: penalty += 1e6
            if c[i,0] - r[i] < 0: penalty += 1e6 * (r[i] - c[i,0])
            if c[i,0] + r[i] > 1: penalty += 1e6 * (c[i,0] + r[i] - 1)
            if c[i,1] - r[i] < 0: penalty += 1e6 * (r[i] - c[i,1])
            if c[i,1] + r[i] > 1: penalty += 1e6 * (c[i,1] + r[i] - 1)
            
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((c[i] - c[j])**2))
                if dist < r[i] + r[j]:
                    overlap = r[i] + r[j] - dist
                    penalty += 1e6 * overlap
        return score - penalty

    # Initial score
    current_score = get_validity_score(current_centers, current_radii)
    
    # If score is bad (penalty), we need to shrink radii or move centers.
    # Let's reset radii to be valid first.
    # Calculate max valid radius for each circle given others fixed?
    # Iterative relaxation.
    
    for _ in range(100):
        changed = False
        for i in range(n):
            max_r = min(current_centers[i,0], 1-current_centers[i,0], 
                        current_centers[i,1], 1-current_centers[i,1])
            for j in range(n):
                if i == j: continue
                dist = np.sqrt(np.sum((current_centers[i] - current_centers[j])**2))
                # dist >= r_i + r_j => r_i <= dist - r_j
                max_r = min(max_r, dist - current_radii[j])
            # If max_r is negative, we have a problem with current_centers (too close)
            # But we assume centers are valid initially.
            # If radii are too big, we clamp.
            if current_radii[i] > max_r + 1e-6:
                current_radii[i] = max(0, max_r)
                changed = True
        if not changed: break

    current_score = get_validity_score(current_centers, current_radii)
    best_score = current_score
    best_centers = current_centers.copy()
    best_radii = current_radii.copy()
    
    # Local Search
    # Try to move centers or increase radii to improve score.
    # Since we want to maximize sum of radii, we can try to increase radii.
    # But increasing radii causes overlaps. So we must move centers.
    
    step_size = 0.01
    for iter in range(2000):
        # Randomly select a circle to move or resize
        idx = random.randint(0, n-1)
        
        # Option 1: Move center
        # Option 2: Increase radius
        
        action = random.random()
        
        if action < 0.7: # 70% chance to move center
            dx = (random.random() - 0.5) * step_size
            dy = (random.random() - 0.5) * step_size
            new_c = current_centers[idx].copy()
            new_c[0] += dx
            new_c[1] += dy
            
            # Check if move is valid with current radii?
            # Or just accept and let penalty handle it?
            # Let's try to accept if it improves score.
            
            # Try move
            old_c = current_centers[idx].copy()
            current_centers[idx] = new_c
            new_score = get_validity_score(current_centers, current_radii)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
            else:
                # Revert
                current_centers[idx] = old_c
        else:
            # 30% chance to increase radius
            # Try to increase r by small amount
            expansion = random.random() * step_size * 0.5
            new_r = current_radii[idx] + expansion
            
            # Clamp to boundary constraints roughly
            # But let penalty handle strict constraints
            # Actually, better to clamp to wall distance to avoid huge penalties
            wall_limit = min(new_c[0] if 'new_c' in locals() else current_centers[idx,0], 
                             1 - (new_c[0] if 'new_c' in locals() else current_centers[idx,0]),
                             (new_c[1] if 'new_c' in locals() else current_centers[idx,1]),
                             1 - (new_c[1] if 'new_c' in locals() else current_centers[idx,1]))
            
            # Note: 'new_c' might not be defined if we didn't move.
            cx, cy = current_centers[idx]
            wall_limit = min(cx, 1-cx, cy, 1-cy)
            
            if new_r > wall_limit:
                new_r = wall_limit # Hard constraint for walls
            
            old_r = current_radii[idx]
            current_radii[idx] = new_r
            new_score = get_validity_score(current_centers, current_radii)
            
            if new_score > current_score:
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
                    best_radii = current_radii.copy()
            else:
                # Revert
                current_radii[idx] = old_r

        # Decay step size
        if iter % 500 == 0:
            step_size *= 0.9
            
    # Final validation and clean up
    # The best configuration might have tiny violations due to penalty function.
    # We need to project to a valid state.
    # If best_score is high and close to sum(radii), it's likely valid.
    # Let's check validity explicitly.
    
    centers_final = best_centers
    radii_final = best_radii
    
    # Check for any minor overlaps and shrink radii to fix them
    # This ensures strict validity for the validator.
    # We can iteratively shrink the smallest overlapping radius?
    # Or just shrink all radii by a small epsilon if needed.
    
    # Let's run a "fix overlaps" pass.
    # Sort circles by radius descending?
    # Actually, just check all pairs. If overlap, reduce radii of both slightly?
    # Better: reduce radius of the smaller circle?
    # Or just uniformly scale down radii if overlaps exist.
    
    # Simple fix:
    for _ in range(10): # Passes
        max_overlap = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers_final[i] - centers_final[j])**2))
                if dist < radii_final[i] + radii_final[j] - 1e-12:
                    overlap = radii_final[i] + radii_final[j] - dist
                    # Reduce radii proportionally?
                    # Or just reduce both by overlap/2
                    radii_final[i] -= overlap / 2.0
                    radii_final[j] -= overlap / 2.0
                    # Ensure non-negative
                    radii_final[i] = max(0, radii_final[i])
                    radii_final[j] = max(0, radii_final[j])
                    max_overlap = max(max_overlap, overlap)
        
        # Check wall constraints
        for i in range(n):
            x, y = centers_final[i]
            r = radii_final[i]
            # Adjust radius to fit in walls
            r_wall = min(x, 1-x, y, 1-y)
            if r > r_wall + 1e-12:
                radii_final[i] = max(0, r_wall)
                
        if max_overlap < 1e-12:
            break
            
    # Final check with validator logic (mental)
    # Return
    sum_radii = np.sum(radii_final)
    
    return centers_final, radii_final, sum_radii
