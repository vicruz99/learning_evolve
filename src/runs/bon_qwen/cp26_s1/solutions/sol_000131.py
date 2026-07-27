# sol_000131 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0b92a944) state=9b8736c2 sum of radii=2.323689 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import random

def run_packing():
    # Set seed for reproducibility
    np.random.seed(42)

    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # --- Step 1: Initial Hexagonal Packing ---
    # Try to fit 26 circles in a hexagonal pattern.
    # Rows of 5 and 4 alternating often works well.
    # 5 + 4 + 5 + 4 + 5 + 4 = 27 (Too many)
    # 5 + 5 + 5 + 5 + 5 + 1 = 26 (Maybe)
    # 5 + 4 + 5 + 4 + 5 + 3 = 26
    
    # Let's try a pattern that sums to 26.
    # 5, 5, 5, 5, 5, 1 is risky for the 1.
    # 5, 5, 5, 5, 4, 2?
    # Let's stick to a dense hexagonal grid approach.
    # 6 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 1 circle? No, better to distribute.
    
    # Let's try 5, 5, 5, 5, 5, 1 arrangement but optimized.
    # Or maybe 4 rows of 6 and 2 rows of 1?
    # Width constraint for 6 circles: 12r <= 1 -> r <= 0.083. Sum ~ 2.16.
    # Width constraint for 5 circles: 10r <= 1 -> r <= 0.1.
    # So 5 per row is better for radius.
    
    # Let's try 5 rows of 5 (25 circles) + 1 small circle in a gap.
    # But we want to maximize sum. Maybe unequal radii.
    
    # Better approach: Random start near a grid, then optimize.
    # Grid 5x5 is tight. Let's perturb a 5x5 grid + 1 extra circle.
    
    # Generate 26 initial positions
    # 5x5 grid centers:
    # x: 0.1, 0.3, 0.5, 0.7, 0.9
    # y: 0.1, 0.3, 0.5, 0.7, 0.9
    # This fits 25 circles of radius 0.1.
    
    # We need 26. Let's place 25 in a slightly smaller grid and 1 in the middle or a gap.
    # Or just random initialization in [0,1] with small radii and let optimizer grow them.
    
    # Initialization:
    # Place circles in a hexagonal pattern scaled to fit.
    # Target radius ~ 0.1.
    
    r_init = 0.085
    centers_list = []
    
    # Hexagonal packing logic
    # Row 0: 5 circles
    # Row 1: 4 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 5 circles
    # Row 5: 3 circles (shifted) -> Total 26
    
    # Vertical spacing dy = r * sqrt(3)
    # Horizontal spacing dx = 2 * r
    
    # Let's construct coordinates
    row_counts = [5, 4, 5, 4, 5, 3]
    idx = 0
    
    # Approximate scaling to fit in 1x1
    # Max width for 5 circles: 10*r. If r=0.085, width=0.85.
    # Max height for 6 rows: 2*r + 5*dy. 
    # dy = 0.085 * 1.732 = 0.147. 
    # Height = 0.17 + 5*0.147 = 0.17 + 0.735 = 0.905. Fits.
    
    for i, count in enumerate(row_counts):
        y = r_init + i * (r_init * np.sqrt(3))
        # Shift odd rows
        x_start = r_init
        if i % 2 == 1:
            x_start = r_init * 2 # Shift by r? No, shift by r implies center at 2r. 
                                 # Distance to prev row center (at r) is sqrt(r^2 + dy^2) = 2r.
                                 # So shift is r. x_start should be r_init + r_init = 2*r_init.
        
        for j in range(count):
            x = x_start + j * (2 * r_init)
            centers_list.append([x, y])
            idx += 1
            
    # If we didn't get 26, adjust. 5+4+5+4+5+3 = 26. Good.
    
    centers = np.array(centers_list)
    radii = np.full(n, r_init)
    
    # --- Step 2: Optimization using repulsion and expansion ---
    
    # We will iterate to increase radii and fix positions.
    # This is a heuristic "force-directed" graph layout.
    
    learning_rate = 0.01
    expansion_rate = 1.0005 # Slightly increase radii each step
    
    for step in range(2000):
        # Increase radii
        radii *= expansion_rate
        # Cap radii at 0.2 just in case
        radii = np.clip(radii, 0, 0.2)
        
        forces = np.zeros_like(centers)
        
        # Boundary forces (push inward if too close)
        # Walls at 0 and 1.
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (x - r) * 10 # Push right
            # Right wall
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 10 # Push left
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (y - r) * 10
            # Top wall
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 10
        
        # Circle-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dist_vec = centers[i] - centers[j]
                dist = np.linalg.norm(dist_vec)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    # Repulsion force proportional to overlap
                    # Direction is along dist_vec
                    force_mag = overlap * 0.5 # Spring constant
                    force_vec = (dist_vec / dist) * force_mag
                    forces[i] += force_vec
                    forces[j] -= force_vec
                elif dist == 0:
                    # Handle zero distance by random push
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01

        # Update positions
        centers += forces * learning_rate
        
        # Ensure centers are within bounds (clamping helps stability)
        # But force should handle it. Let's clamp to [r, 1-r] effectively?
        # Just clamp to [0,1] to prevent explosion
        centers = np.clip(centers, 0, 1)
        
        # Adaptive learning rate?
        if step % 500 == 0:
            learning_rate *= 0.9

    # --- Step 3: Local Optimization (Coordinate Descent / Scipy) ---
    # The above might leave circles slightly overlapping or not tight.
    # We can try to solve for maximum r such that constraints hold.
    # But with unequal radii, it's complex.
    # Let's try to optimize positions for fixed radii to reduce overlap, 
    # then increase radii.
    
    # Actually, a better way is to use scipy.optimize to minimize a penalty function.
    # Objective: minimize -sum(radii) (i.e. maximize sum)
    # Variables: centers (n,2), radii (n)
    # But this is high dimensional and non-convex.
    
    # Let's refine the current solution.
    # Check for overlaps and push them apart.
    
    for _ in range(100):
        # Check overlaps
        has_overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                min_dist = radii[i] + radii[j]
                if dist < min_dist - 1e-9:
                    has_overlap = True
                    # Push apart
                    overlap = min_dist - dist
                    # Move i away from j, j away from i
                    # We can just scale positions relative to midpoint?
                    # Or simply apply a force.
                    vec = centers[i] - centers[j]
                    if dist > 1e-9:
                        shift = (vec / dist) * (overlap / 2)
                        centers[i] += shift
                        centers[j] -= shift
                    else:
                        centers[i] += np.random.randn(2) * 0.001
                        centers[j] -= np.random.randn(2) * 0.001
        
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0: centers[i, 0] = r
            if x + r > 1: centers[i, 0] = 1 - r
            if y - r < 0: centers[i, 1] = r
            if y + r > 1: centers[i, 1] = 1 - r
            
        if not has_overlap:
            # Try to increase radii slightly
            # Find max possible increase?
            # Simple greedy increase
            increase = 1.001
            radii *= increase
            
            # Re-check and fix overlaps if any created
            # (The loop above will handle it in next iteration)

    # --- Step 4: Final Polish ---
    # Ensure strict validity
    # Re-run a strict overlap check and adjustment
    
    # Use a small optimization step to maximize radii sum
    # We can treat this as: for current centers, what is the max uniform scaling?
    # But radii are not uniform.
    
    # Let's just run the validation to see if we are close.
    # If overlaps exist, reduce radii.
    
    # Actually, the loop above might have increased radii too much.
    # Let's reduce radii until valid.
    
    # Binary search for a scaling factor k for radii
    # Current radii might be slightly invalid.
    # Find max k such that radii * k is valid.
    
    low, high = 0.0, 1.0
    # Check if current is valid
    is_valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-7 or x + r > 1 + 1e-7 or y - r < -1e-7 or y + r > 1 + 1e-7:
            is_valid = False
            break
    if is_valid:
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-7:
                    is_valid = False
                    break
            if not is_valid: break
            
    if not is_valid:
        # Scale down radii
        # Just a simple reduction
        factor = 0.95
        while not is_valid:
            radii *= factor
            # Check validity again
            is_valid = True
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x - r < -1e-7 or x + r > 1 + 1e-7 or y - r < -1e-7 or y + r > 1 + 1e-7:
                    is_valid = False
                    break
            if is_valid:
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.linalg.norm(centers[i] - centers[j])
                        if dist < radii[i] + radii[j] - 1e-7:
                            is_valid = False
                            break
                    if not is_valid: break
            factor *= 0.95 # Reduce step
            
    # One last attempt to increase radii
    # Try to increase all radii by a tiny amount if valid
    # But with different radii, it's tricky.
    # Let's just rely on the simulation.
    
    # However, to be safe, let's ensure no NaNs and valid bounds.
    centers = np.clip(centers, 0, 1)
    radii = np.abs(radii)
    
    # Calculate sum
    sum_radii = np.sum(radii)
    
    # If sum is very low, maybe initialization failed?
    # With 26 circles, sum ~ 2.5 is expected.
    
    return centers, radii, sum_radii

# Let's refine the strategy to be more robust.
# The random initialization might be poor.
# Let's force a structured initialization.

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Heuristic for 26 circles:
    # A common optimal packing for n=26 involves a mix of rows.
    # Let's try to construct a valid packing with r approx 0.1.
    # 5 rows of 5 circles is 25.
    # Where to put 26th?
    # Maybe shift the grid to make space.
    
    # Let's use the "Apollonian gasket" idea or just dense packing.
    # But simpler:
    # Start with a hexagonal lattice.
    
    # We want to maximize sum of radii.
    # Let's try to place circles at specific coordinates known to work or optimize.
    
    # Initial configuration:
    # 6 rows.
    # Row 0: 5 circles
    # Row 1: 4 circles
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 5 circles
    # Row 5: 3 circles
    # Total 26.
    
    # Parameters
    r = 0.095 # Start slightly less than 0.1
    dy = r * np.sqrt(3)
    dx = 2 * r
    
    count = 0
    row_counts = [5, 4, 5, 4, 5, 3]
    
    for i, cnt in enumerate(row_counts):
        y = r + i * dy
        x_start = r
        if i % 2 == 1:
            x_start = 2 * r # Shifted row
        
        for j in range(cnt):
            x = x_start + j * dx
            centers[count] = [x, y]
            radii[count] = r
            count += 1
            
    # Now we have a valid packing (likely, since r=0.095 is small).
    # We want to increase r.
    # We can scale radii up until constraints are hit.
    
    # Let's find the limiting factor.
    # 1. Boundaries
    # 2. Overlaps
    
    # We can increase r uniformly?
    # If we increase r uniformly, positions stay same, but circles grow.
    # Overlaps will occur first between nearest neighbors.
    # Nearest neighbor distance in hex grid is 2r (initially).
    # If we scale r by k, dist stays same (centers fixed), but required dist becomes 2kr.
    # So we are limited by current dist / (2 * r_old) ?
    # Actually, if we just scale radii, we don't move centers.
    # The "tightest" constraint is the minimum distance between centers / sum of radii.
    # Currently dist = 2r, sum = 2r. Ratio = 1.
    # So we can't increase r without moving centers.
    
    # So we need to move centers apart.
    # We can expand the configuration.
    # Scale centers relative to center of square (0.5, 0.5)?
    # Or just optimize.
    
    # Let's use the force-directed approach from before but starting from this valid config.
    
    learning_rate = 0.05
    expansion_factor = 1.0002
    
    for step in range(3000):
        radii *= expansion_factor
        
        forces = np.zeros_like(centers)
        
        # Boundary
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0: forces[i, 0] += (r - x) * 5 # Push out? No, x-r < 0 means x < r. Push right.
            # Correction: if x - r < 0, circle is cut by left wall. Push center right.
            # Force direction +x. Magnitude proportional to penetration.
            penetration = r - x
            forces[i, 0] += penetration * 10
            
            if x + r > 1: 
                penetration = x + r - 1
                forces[i, 0] -= penetration * 10
                
            if y - r < 0:
                penetration = r - y
                forces[i, 1] += penetration * 10
                
            if y + r > 1:
                penetration = y + r - 1
                forces[i, 1] -= penetration * 10

        # Repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-10:
                    overlap = min_dist - dist
                    force = overlap * 2.0 # Stiffness
                    fx = (dx / dist) * force
                    fy = (dy / dist) * force
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy

        centers += forces * learning_rate
        # Clip to keep inside
        # But force should keep them in.
        # However, numerical errors.
        # Just clamp radii to ensure they fit if we are stuck?
        # No, let forces work.
        
        if step % 100 == 0:
            learning_rate *= 0.95

    # Post-processing: ensure strict validity
    # Check for overlaps and reduce radii if necessary
    for i in range(n):
        # Boundary check
        x, y = centers[i]
        r = radii[i]
        # Adjust radius to fit boundary
        max_r_x = min(x, 1-x)
        max_r_y = min(y, 1-y)
        radii[i] = min(radii[i], max_r_x, max_r_y)
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            sum_r = radii[i] + radii[j]
            if dist < sum_r:
                # Reduce radii to fit
                # Distribute reduction
                diff = sum_r - dist
                # Reduce both by half
                radii[i] -= diff / 2
                radii[j] -= diff / 2
                # Ensure non-negative
                radii[i] = max(0, radii[i])
                radii[j] = max(0, radii[j])
    
    # Re-check boundary after reducing radii for overlaps?
    # Radii reduced, so boundary constraint looser. OK.
    
    # Final check and fix
    # If any radius is negative (shouldn't be), fix.
    radii = np.maximum(radii, 0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Wait, the reduction loop might be slow or iterative.
# But n=26 is small.
# However, reducing radii[i] and [j] might cause [j] to overlap with [k].
# So we might need multiple passes.
# But for the purpose of this problem, a single pass reduction is usually enough to get a valid packing,
# though not optimal.
# But since we expanded radii during simulation, we are likely close to optimal.

# Let's refine the reduction to be robust.
def adjust_for_validity(centers, radii):
    n = centers.shape[0]
    # Boundary
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        r = min(r, x, 1-x, y, 1-y)
        radii[i] = r
    
    # Overlaps
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                s = radii[i] + radii[j]
                if s > dist:
                    # Reduce
                    diff = s - dist
                    # Reduce proportionally? Or equally?
                    # Equally is simpler.
                    r1, r2 = radii[i], radii[j]
                    radii[i] -= diff / 2
                    radii[j] -= diff / 2
                    radii[i] = max(0, radii[i])
                    radii[j] = max(0, radii[j])
                    changed = True
    return radii

# Final function structure
def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Initialization
    r = 0.09
    dy = r * np.sqrt(3)
    dx = 2 * r
    count = 0
    row_counts = [5, 4, 5, 4, 5, 3]
    
    for i, cnt in enumerate(row_counts):
        y = r + i * dy
        x_start = r
        if i % 2 == 1:
            x_start = 2 * r
        for j in range(cnt):
            x = x_start + j * dx
            centers[count] = [x, y]
            radii[count] = r
            count += 1
            
    # Optimization loop
    lr = 0.02
    for step in range(2000):
        radii *= 1.0001
        forces = np.zeros_like(centers)
        
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Boundary
            if x < r: forces[i, 0] += (r - x) * 5
            if x > 1 - r: forces[i, 0] -= (x - (1 - r)) * 5
            if y < r: forces[i, 1] += (r - y) * 5
            if y > 1 - r: forces[i, 1] -= (y - (1 - r)) * 5
            
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_d = radii[i] + radii[j]
                if dist < min_d and dist > 1e-12:
                    overlap = min_d - dist
                    f = overlap * 1.0
                    fx = (dx / dist) * f
                    fy = (dy / dist) * f
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        centers += forces * lr
        if step % 200 == 0:
            lr *= 0.9

    radii = adjust_for_validity(centers, radii)
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
