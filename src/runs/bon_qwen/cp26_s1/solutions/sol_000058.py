# sol_000058 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2252d37f) state=4d292033 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def dist(c1, c2):
    return np.sqrt(np.sum((c1 - c2) ** 2))

def get_max_radius_for_center(c, centers, radii, exclude_idx=-1):
    """
    Calculate the maximum possible radius for a circle at center c
    given the positions of other circles.
    """
    r_max = 1.0 # Upper bound by square size (actually 0.5)
    
    # Boundary constraints
    r_max = min(c[0], 1.0 - c[0], c[1], 1.0 - c[1])
    
    # Overlap constraints
    for i in range(len(centers)):
        if i == exclude_idx:
            continue
        d = dist(c, centers[i])
        # The radius r must satisfy r + radii[i] <= d  => r <= d - radii[i]
        r_lim = d - radii[i]
        if r_lim < r_max:
            r_max = r_lim
            
    return max(0.0, r_max)

def run_packing():
    n = 26
    np.random.seed(42)
    
    # 1. Initialization: Perturbed Grid
    # A 5x5 grid fits 25 circles of radius 0.1 exactly.
    # To fit 26, we need to shrink slightly or rearrange.
    # Let's try to initialize with a layout that might lead to unequal radii.
    # We'll start with a grid of 25 circles with radius 0.09, and place the 26th in the center.
    
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # 5x5 grid centers
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    
    idx = 0
    for y in grid_y:
        for x in grid_x:
            centers[idx] = [x, y]
            radii[idx] = 0.09 # Initial radius
            idx += 1
            
    # Place 26th circle in the center gap
    centers[idx] = [0.5, 0.5]
    radii[idx] = 0.05
    
    # 2. Optimization Loop
    # We will iteratively try to increase radii and resolve conflicts.
    
    num_iterations = 1000
    expansion_rate = 0.005 # How much to try to grow radii
    
    for it in range(num_iterations):
        # Try to expand radii
        # We expand by a small factor or add a small amount.
        # Since we want to maximize sum, growing is good.
        # But we must maintain validity.
        
        # Strategy: Increase radii slightly, then push circles apart to satisfy constraints.
        
        # Increase radii
        radii += expansion_rate
        
        # Resolve overlaps and boundary violations
        # We iterate a few times to let the system settle
        for _ in range(5):
            moved = False
            for i in range(n):
                ci = centers[i]
                ri = radii[i]
                
                # Check boundary
                # If circle i is too close to boundary, move it towards center
                # Actually, to maximize radius, it should be as far from boundary as possible?
                # No, if it's touching boundary, radius is limited by distance to boundary.
                # Moving towards center increases distance to boundary, allowing larger radius.
                # But it might cause overlap with others.
                # However, usually circles are pushed outwards by neighbors.
                
                # Check boundary constraints for current position
                min_dist_to_wall = min(ci[0], 1.0 - ci[0], ci[1], 1.0 - ci[1])
                if ri > min_dist_to_wall + 1e-9:
                    # Circle is outside or too big for position
                    # Move center to be at distance ri from closest wall
                    # This is tricky. If it's stuck, we might need to shrink radius.
                    # But let's try to move it to the center of the valid region for this radius?
                    # Actually, simply clamping position might work if we assume radius is fixed.
                    # But radius is variable.
                    pass # Handled by overlap resolution mostly?
                
                # Check overlaps
                for j in range(i + 1, n):
                    cj = centers[j]
                    rj = radii[j]
                    
                    d = dist(ci, cj)
                    min_d = ri + rj
                    
                    if d < min_d - 1e-9:
                        # Overlap detected. Push apart.
                        # Displacement vector
                        if d < 1e-9:
                            dx, dy = 1.0, 0.0 # Random push if coincident
                        else:
                            dx = (ci[0] - cj[0]) / d
                            dy = (ci[1] - cj[1]) / d
                        
                        overlap = min_d - d
                        # Move each circle half the overlap distance away from each other
                        # To maintain symmetry or balance
                        move = overlap / 2.0
                        
                        new_ci = ci + np.array([dx, dy]) * move
                        new_cj = cj - np.array([dx, dy]) * move
                        
                        # Check if new positions are valid (inside square)
                        # If not, we might need to adjust differently or shrink radii
                        # But for now, just clamp to square?
                        
                        # Simple projection back to square
                        new_ci = np.clip(new_ci, radii[i], 1.0 - radii[i])
                        new_cj = np.clip(new_cj, radii[j], 1.0 - radii[j])
                        
                        # Re-calculate distance after move/clamp to check if resolved?
                        # Ideally, we update centers
                        centers[i] = new_ci
                        centers[j] = new_cj
                        moved = True

        # Additional step: Optimize positions for current radii?
        # If we can't increase radii anymore, we are stuck.
        # We can try to swap or random perturbations to escape local minima.
        if it % 100 == 0:
            # Random perturbation
            for i in range(n):
                perturbation = np.random.uniform(-0.01, 0.01, 2)
                new_pos = centers[i] + perturbation
                # Check if new pos allows current radius (boundary)
                if 0 <= new_pos[0] <= 1 and 0 <= new_pos[1] <= 1:
                    # Check overlaps roughly?
                    # Just move if it doesn't immediately violate boundary
                    # Overlaps will be fixed in next iterations
                    centers[i] = new_pos

    # 3. Final Adjustment: Shrink radii to strictly satisfy constraints if needed
    # And ensure circles are inside square.
    # The optimization above pushed circles but might have left small violations due to discretization.
    # We perform a final "tightening" pass.
    
    # For each circle, calculate max possible radius given fixed centers
    # Then set radius to that max (or current if smaller, but we want to maximize sum)
    # Wait, if we fix centers, we can increase radii.
    # But centers are not fixed, they are variables.
    # However, at the end of optimization, we can assume a configuration.
    # Let's recalculate valid radii for the final centers to ensure validity.
    
    final_radii = np.zeros(n)
    for i in range(n):
        # Max radius based on boundaries
        r_bound = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        r_bound = max(0.0, r_bound)
        
        # Max radius based on neighbors
        r_min = r_bound
        for j in range(n):
            if i == j: continue
            d = dist(centers[i], centers[j])
            # We need r_i + r_j <= d => r_i <= d - r_j
            # But r_j is also a variable. This is coupled.
            # A simple fix: assume current radii are close to optimal, 
            # just ensure no overlap by reducing radius if necessary.
            # But better: solve for valid radii?
            # If we just want a valid packing, we can set r_i = min(r_bound, min(d - r_j))
            # but r_j depends on r_i.
            pass 
        
        # Let's just use the optimized radii, but clamp to boundaries
        if centers[i, 0] < radii[i]: radii[i] = centers[i, 0]
        if centers[i, 0] > 1 - radii[i]: radii[i] = 1 - centers[i, 0]
        if centers[i, 1] < radii[i]: radii[i] = centers[i, 1]
        if centers[i, 1] > 1 - radii[i]: radii[i] = 1 - centers[i, 1]
        radii[i] = max(0.0, radii[i])

    # Resolve remaining overlaps by shrinking radii slightly if needed
    # Since the optimization loop was aggressive, there might be tiny overlaps.
    # We can iteratively shrink the smaller circle in an overlap pair.
    for _ in range(50):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                d = dist(centers[i], centers[j])
                if d < radii[i] + radii[j] - 1e-10:
                    # Shrink the larger radius or split difference
                    diff = radii[i] + radii[j] - d
                    shrink = diff / 2.0
                    # Shrink radii
                    radii[i] -= shrink
                    radii[j] -= shrink
                    overlap_found = True
                    # Ensure non-negative
                    if radii[i] < 0: radii[i] = 0
                    if radii[j] < 0: radii[j] = 0
        if not overlap_found:
            break
            
    # Final validation and cleanup
    # Ensure centers are strictly inside [0,1] considering radii
    for i in range(n):
        r = radii[i]
        centers[i, 0] = max(r, min(1.0 - r, centers[i, 0]))
        centers[i, 1] = max(r, min(1.0 - r, centers[i, 1]))
        
        # Re-check radius against boundaries after clamping center
        # If center was clamped, radius might need to be smaller?
        # Actually if center is clamped to [r, 1-r], it's valid.
        # But if r was large, center might have been forced to valid range.
        # We should ensure r <= distance to boundary.
        dist_x = min(centers[i, 0], 1.0 - centers[i, 0])
        dist_y = min(centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], dist_x, dist_y)
        radii[i] = max(0.0, radii[i])

    sum_radii = np.sum(radii)
    
    # Verify
    # if not validate_packing(centers, radii):
    #     print("Warning: Final packing failed validation. Attempting fix...")
    #     # Fallback to simple grid if optimization failed?
    #     pass

    return centers, radii, sum_radii
