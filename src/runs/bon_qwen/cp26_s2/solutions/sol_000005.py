# sol_000005 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f294fc76) state=f7acbbd6 sum of radii=1.318944 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_radii: float sum of all radii
    """
    n_circles = 26
    np.random.seed(42)  # For reproducibility

    # 1. Initialization: Hexagonal Grid
    # We arrange circles in a hexagonal pattern to start with a dense packing.
    # We'll try to fit them in a 5x6 area roughly.
    rows = 5
    cols = 6
    # Radius estimate for 5x5 equal circles is 0.1. 
    # We start smaller to allow movement.
    initial_r = 0.04
    
    centers = []
    radii = []
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= n_circles:
                break
            # Hexagonal offset
            x = 0.05 + c * 0.15 + (r % 2) * 0.075
            y = 0.05 + r * 0.13
            centers.append([x, y])
            radii.append(initial_r)
            count += 1
        if count >= n_circles:
            break
    
    centers = np.array(centers)
    radii = np.array(radii)

    # 2. Optimization Loop
    # We will iteratively increase radii and resolve conflicts using repulsion.
    
    n_iters = 2000
    # Decay rate for radius growth
    growth_rate = 0.0005
    # Repulsion strength
    repulsion_strength = 0.5
    # Damping factor for position updates
    damping = 0.9

    # We'll store velocities for the centers
    velocity = np.zeros_like(centers)

    for iteration in range(n_iters):
        # Adaptive growth rate: slow down as we progress
        current_growth = growth_rate * (1.0 - iteration / n_iters)
        current_growth = max(current_growth, 1e-6)

        # Try to increase all radii
        radii += current_growth

        # Check constraints and apply forces
        # Forces accumulate for this iteration
        forces = np.zeros_like(centers)

        # 1. Boundary constraints
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                radii[i] = min(radii[i], x) # Shrink if too big for position
                forces[i, 0] += (0 - (x - radii[i])) * 10 # Push right
            # Right wall
            if x + r > 1:
                radii[i] = min(radii[i], 1 - x)
                forces[i, 0] -= (1 - (x + radii[i])) * 10 # Push left
            # Bottom wall
            if y - r < 0:
                radii[i] = min(radii[i], y)
                forces[i, 1] += (0 - (y - radii[i])) * 10 # Push up
            # Top wall
            if y + r > 1:
                radii[i] = min(radii[i], 1 - y)
                forces[i, 1] -= (1 - (y + radii[i])) * 10 # Push down

        # 2. Overlap constraints
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist_vec = centers[j] - centers[i]
                dist = np.linalg.norm(dist_vec)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap detected
                    # Normalize direction
                    dir_vec = dist_vec / dist
                    
                    # Calculate penetration depth
                    penetration = min_dist - dist
                    
                    # Apply repulsive force proportional to penetration
                    # Stronger repulsion for smaller distances to prevent jamming
                    force_mag = penetration * repulsion_strength
                    
                    # Adjust radii if stuck? 
                    # If circles are heavily overlapping, we might need to shrink one.
                    # But primarily we push them apart.
                    
                    # To maximize sum of radii, we prefer to keep radii large.
                    # However, if they overlap, we must separate or shrink.
                    # Let's try to separate first. If separation is impossible (walls), 
                    # the boundary checks above will shrink them.
                    
                    forces[i] -= dir_vec * force_mag
                    forces[j] += dir_vec * force_mag
                    
                    # Optional: Shrink the smaller circle slightly if overlap is severe
                    # This helps escape local minima where two circles trap each other
                    if penetration > 1e-4:
                        # Shrink the one with smaller radius less? 
                        # Actually, shrinking reduces objective. 
                        # We rely on forces to move them.
                        pass

        # Update velocities and positions
        velocity = velocity * damping + forces
        centers += velocity
        
        # Clip centers to stay within reasonable bounds (0 to 1)
        # Although forces should handle this, numerical errors can occur.
        # We ensure centers are not outside [r, 1-r] approximately, 
        # but the constraint check above shrinks radii if centers violate.
        # Let's just clamp centers to [0, 1] to prevent explosion.
        centers = np.clip(centers, 0, 1)

        # Re-verify radii against current center positions (hard constraints)
        # If a center is at 0.05 and radius tries to be 0.1, shrink radius.
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            max_r = min(x, 1-x, y, 1-y)
            if radii[i] > max_r:
                radii[i] = max_r

    # Final validation and cleanup
    # Ensure no negative radii
    radii = np.maximum(radii, 0)
    
    # Calculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
