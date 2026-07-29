# sol_000324 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fe3e1745) state=8e040b8b sum of radii=0.312073 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed expansion algorithm starting from a hexagonal lattice.
    """
    n_circles = 26
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None

    # Number of restarts to avoid local minima
    num_restarts = 10
    
    for restart in range(num_restarts):
        centers = np.zeros((n_circles, 2))
        
        # 1. Initialization: Hexagonal Grid
        # Estimate radius for 26 circles. 
        # Area approx 1. Density ~ 0.9. 26 * pi * r^2 approx 0.9 => r approx 0.105
        # Let's start with a grid placement.
        # We want to place 26 points. A 5x6 grid has 30 points. 
        # We can place them in a dense formation.
        
        # Hexagonal packing parameters
        # Let's try to fit them in a pattern.
        # Row spacing dy = sqrt(3)/2 * 2r = sqrt(3)*r. Col spacing dx = 2r.
        # Approximate r ~ 0.1. dx ~ 0.2, dy ~ 0.173.
        # Square size 1. 
        # Cols ~ 1/0.2 = 5. Rows ~ 1/0.173 = 5.7.
        # 5x6 = 30 points. We need 26.
        
        # Let's generate a hexagonal grid and pick the first 26 or place them smartly.
        # Actually, just placing them in a slightly perturbed grid is good.
        
        # Random perturbation for diversity in restarts
        np.random.seed(42 + restart)
        
        # Grid placement
        # 6 columns, roughly 5 rows?
        # Let's try to distribute them evenly.
        # sqrt(26) approx 5.1. So 5x5 or 6x5.
        
        # Create a list of grid points
        # We'll use a hexagonal layout generator
        points = []
        r_est = 0.1 # Initial estimate
        
        # Rows
        y = r_est
        row_idx = 0
        while y < 1 - r_est:
            # Cols
            # Even rows start at r_est, odd rows shifted by r_est (or dx/2)
            # In hex packing, horizontal shift is r_est (since dx = 2r_est, half is r_est)
            # Actually, if centers are at x, x+2r, next row is at x+r, x+3r...
            
            start_x = r_est
            if row_idx % 2 == 1:
                start_x = 2 * r_est # Shift by one radius? No, shift by r_est?
                # If row 0: r, 3r, 5r...
                # Row 1: 2r, 4r... -> shift is r.
                # Wait, distance between (r,r) and (2r, r+sqrt(3)r)
                # dx = r, dy = sqrt(3)r. dist^2 = r^2 + 3r^2 = 4r^2. dist = 2r. Correct.
                # So shift is r_est.
                start_x = 2 * r_est 
                # Wait, if row 0 starts at r_est, next is 3*r_est.
                # Row 1 should be at 2*r_est? 
                # (r_est, r_est) to (2*r_est, r_est + sqrt(3)*r_est).
                # dx = r_est. dy = sqrt(3)*r_est.
                # dist = 2*r_est. Yes.
                # So if row 0 x's are r, 3r, 5r...
                # Row 1 x's are 2r, 4r, 6r...
                # So start_x for odd rows is 2*r_est.
                # But wait, if we use step 2r, row 0 is r, 3r.
                # Row 1 starts at 2r?
                # Distance from 3r to 2r is r. Yes.
                pass
            
            x = start_x
            while x < 1 - r_est:
                if len(points) < n_circles:
                    points.append([x, y])
                x += 2 * r_est
            y += math.sqrt(3) * r_est
            row_idx += 1
        
        # If we didn't get 26 points (unlikely with r=0.1), fallback to random
        if len(points) < n_circles:
            # Fallback: random grid
            cols = int(np.ceil(np.sqrt(n_circles)))
            rows = int(np.ceil(n_circles / cols))
            idx = 0
            for r in range(rows):
                for c in range(cols):
                    if idx < n_circles:
                        # Place in grid
                        cx = (c + 0.5) / cols
                        cy = (r + 0.5) / rows
                        points.append([cx, cy])
                        idx += 1
        
        # Take first 26
        init_centers = np.array(points[:n_circles])
        
        # Add small random noise to break symmetry and help escape local minima
        noise_scale = 0.005
        init_centers += noise_scale * np.random.randn(n_circles, 2)
        
        # Clamp to valid range [0, 1]
        init_centers = np.clip(init_centers, 0, 1)
        
        # 2. Optimization: Expanding Circles / Force Directed
        # We want to find max r such that circles of radius r fit.
        # We will iteratively increase r and resolve collisions.
        
        current_centers = init_centers.copy()
        current_r = 0.01 # Start small
        
        # Parameters
        max_r = 0.5
        steps_per_iteration = 100
        num_iterations = 100 # How many times we try to expand r
        r_step = (max_r - current_r) / num_iterations # Not really, we adapt
        
        # Adaptive r increase
        # In each "phase", we try to increase r slightly and relax.
        
        # Let's use a fixed number of relaxation steps and increase r slowly.
        # But simpler: Just run a simulation for a long time where r increases.
        
        # Actually, let's fix r and optimize positions first to find a good config,
        # then binary search? No, simulation is easier.
        
        # Simulation loop
        # We will run for a fixed number of steps.
        # In each step, we compute forces and move centers.
        # We also slowly grow r.
        
        # To maximize sum of radii, we can assume all radii are equal to r.
        # So we just maximize r.
        
        r = current_r
        learning_rate = 0.01
        decay = 0.995
        
        for step in range(2000):
            # Calculate forces
            forces = np.zeros_like(current_centers)
            
            # Pairwise repulsion
            # Vectorized distance calculation
            # dist_sq = sum((c_i - c_j)^2)
            # We need to handle pairs.
            
            # Optimization: use broadcasting
            # centers shape (N, 2)
            # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (N, N, 2)
            # dists = np.sqrt(np.sum(diff**2, axis=2)) # (N, N)
            
            # To save memory/time for N=26, loop is fine or full matrix.
            # 26x26 is small.
            
            diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-9) # avoid div by 0
            
            # Mask for upper triangle (i < j)
            # But we can apply force to both i and j.
            # Force magnitude proportional to overlap.
            # Overlap = 2r - dist. If > 0, repel.
            
            min_dist = 2 * r
            overlap = np.maximum(0, min_dist - dists)
            
            # Direction
            # force direction is along diff.
            # unit vector = diff / dists
            # Avoid division by zero
            safe_dists = np.where(dists > 1e-9, dists, 1.0)
            unit_diff = diff / safe_dists[:, :, np.newaxis]
            
            # Force vector = overlap * unit_diff
            # Sum forces for each circle
            # Sum over j for each i
            # Note: overlap matrix is symmetric, unit_diff is antisymmetric (sort of)
            # Actually diff_ij = - diff_ji. unit_diff_ij = - unit_diff_ji.
            # overlap_ij = overlap_ji.
            # So force_ij = - force_ji.
            # Total force on i = sum_j (overlap_ij * unit_diff_ij)
            
            pair_forces = overlap[:, :, np.newaxis] * unit_diff
            forces += np.sum(pair_forces, axis=1)
            
            # Boundary repulsion
            # If x < r, push right. If x > 1-r, push left.
            # Force magnitude proportional to penetration.
            boundary_margin = 0.0 # strict constraint
            
            # Left wall
            penetrations = current_centers[:, 0] - r
            # If penetration < 0, we are inside wall. Push out.
            # Actually boundary is at 0. Circle extent is x-r.
            # Constraint: x - r >= 0 => x >= r.
            # Violation: r - x.
            left_violation = np.maximum(0, r - current_centers[:, 0])
            forces[:, 0] += left_violation * 10 # Strong force
            
            # Right wall
            # x + r <= 1 => x <= 1 - r
            # Violation: x - (1-r)
            right_violation = np.maximum(0, current_centers[:, 0] - (1 - r))
            forces[:, 0] -= right_violation * 10
            
            # Bottom wall
            down_violation = np.maximum(0, r - current_centers[:, 1])
            forces[:, 1] += down_violation * 10
            
            # Top wall
            up_violation = np.maximum(0, current_centers[:, 1] - (1 - r))
            forces[:, 1] -= up_violation * 10
            
            # Update positions
            # Limit max movement to prevent instability
            max_move = 0.05
            move = forces * learning_rate
            # Clamp move
            max_move_val = np.max(np.abs(move))
            if max_move_val > max_move:
                move *= (max_move / max_move_val)
            
            current_centers += move
            
            # Increase r slowly
            # If we are stable (forces small), we can increase r.
            # But for simplicity, just increase r over time.
            # We want to find max r.
            # Let's increase r by a tiny amount every step if no major overlaps?
            # Or just a fixed schedule.
            # Since we have 2000 steps, let's increase r from 0.05 to 0.15.
            
            # Actually, better strategy:
            # Check if current configuration is valid (no overlaps).
            # If valid, try to increase r.
            # If not valid, keep r and try to resolve overlaps.
            
            # Check max overlap
            current_min_dist = np.min(dists[np.triu_indices(n_circles, k=1)])
            current_boundary_dist = np.min(np.minimum(
                np.minimum(current_centers[:, 0], 1 - current_centers[:, 0]),
                np.minimum(current_centers[:, 1], 1 - current_centers[:, 1])
            ))
            
            # Effective radius supported by current config
            supported_r = np.min([current_min_dist / 2, current_boundary_dist])
            
            # If current r is less than supported, we can increase it.
            if r < supported_r - 1e-5:
                r = min(r * 1.001, supported_r) # Increase r
            else:
                # If we are limited by overlaps/boundary, r stays or decreases?
                # We want to push r up, but geometry limits it.
                # If supported_r < r, we have overlaps. We need to move centers.
                # The forces will do that.
                pass

            # Decay learning rate
            learning_rate *= decay
            
            # Clamp centers to [0, 1] strictly to prevent escape
            # Though boundary forces should handle it.
            current_centers = np.clip(current_centers, 0, 1)
            
            # Recalculate r based on actual constraints to ensure validity?
            # No, r is a parameter we are trying to maximize.
            # But we must ensure at the end r is valid.
            # The loop tries to maintain validity.
        
        # After simulation, recalculate valid radii
        # The circles might not be perfectly equal size or valid.
        # We should compute the actual valid radius for each circle?
        # Or just enforce equal radius r.
        
        # Let's enforce equal radius r based on the final configuration.
        # r_final = min(min_dist/2, min_boundary_dist)
        
        # Compute distances again
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        min_pair_dist = np.min(dists[np.triu_indices(n_circles, k=1)])
        
        # Distance to boundaries
        dists_to_boundary = np.minimum(
            np.minimum(current_centers[:, 0], 1 - current_centers[:, 0]),
            np.minimum(current_centers[:, 1], 1 - current_centers[:, 1])
        )
        min_boundary_dist = np.min(dists_to_boundary)
        
        final_r = min(min_pair_dist / 2, min_boundary_dist)
        
        # Ensure non-negative
        if final_r < 0:
            final_r = 0.0
            
        # Calculate sum
        sum_radii = 26 * final_r
        
        if sum_radii > best_sum_radii:
            best_sum_radii = sum_radii
            best_centers = current_centers.copy()
            best_radii = np.full(n_circles, final_r)
            
        # Small perturbation to help next restart if needed?
        # The noise at start handles this.
        
    return best_centers, best_radii, best_sum_radii
