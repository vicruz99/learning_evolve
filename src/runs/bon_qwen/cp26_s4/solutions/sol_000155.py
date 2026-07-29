# sol_000155 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a1c97a27) state=ad07e564 sum of radii=0.001300 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    n = 26
    
    # 1. Initialize in a hexagonal grid pattern
    # Estimate a radius slightly larger than 0.1 to start
    r_init = 0.105
    
    centers = np.zeros((n, 2))
    
    # Create a hexagonal grid pattern for 26 circles
    # 5 rows, varying number of circles per row
    row_counts = [5, 6, 5, 6, 4] # Sum = 26
    # But 6 circles in a row is very tight. Let's try to balance.
    # A 5x5 grid (25) + 1 is safer to start, then optimize.
    # Let's place 26 points in a distorted hex grid.
    
    idx = 0
    # 5 rows
    # Row y coordinates roughly spaced by sqrt(3)/2 * 2r ~ 1.732r
    # Let's just place them in a 6x5 grid subset
    # Or simpler: Random points? No, structure helps.
    
    # Let's try a specific hex arrangement:
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 6 circles (shifted back?)
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Total 26.
    
    # Adjusting to fit width. 
    # If r ~ 0.101, width for 5 circles is ~ 1.01 (tight).
    # Width for 6 circles is ~ 1.21 (impossible).
    # So we cannot have 6 circles in a straight row if r > 0.083.
    # We must rely on the optimization to tilt/shift them.
    # Let's initialize with 5 rows of 5 circles, and one extra in a gap.
    
    centers = []
    # 5 rows of 5
    for r_idx in range(5):
        y = 0.1 + r_idx * 0.2
        for c_idx in range(5):
            x = 0.1 + c_idx * 0.2
            centers.append([x, y])
            
    # 26th circle in the middle gap
    centers.append([0.5, 0.5])
    
    centers = np.array(centers)
    
    # 2. Force-directed optimization to maximize minimum distance
    # We want to maximize min_dist.
    # Equivalent to minimizing potential energy of repulsion.
    
    learning_rate = 0.005
    num_iterations = 2000
    
    # Convert to float
    centers = centers.astype(float)
    
    for it in range(num_iterations):
        forces = np.zeros_like(centers)
        
        # Repulsion from boundaries
        for i in range(n):
            x, y = centers[i]
            # Push from x=0
            forces[i, 0] += max(0, x - 0.01) * 0 # Soft push?
            # Harder constraint: if close to wall, push away
            dist_x_left = x
            dist_x_right = 1.0 - x
            dist_y_bottom = y
            dist_y_top = 1.0 - y
            
            # If distance is small, apply large force
            min_dist_boundary = min(dist_x_left, dist_x_right, dist_y_bottom, dist_y_top)
            
            # We want min_dist_boundary to be large.
            # Force proportional to (target - current)
            # But we don't know target. 
            # Let's just push away from boundaries if too close to a reference radius?
            # Or just use a penalty: if x < 0.05, push right.
            
            if x < 0.05:
                forces[i, 0] += (0.05 - x)
            if x > 0.95:
                forces[i, 0] -= (x - 0.95)
            if y < 0.05:
                forces[i, 1] += (0.05 - y)
            if y > 0.95:
                forces[i, 1] -= (y - 0.95)
                
        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < 0.001:
                    dist = 0.001
                    dx = np.random.randn() * 0.001
                    dy = np.random.randn() * 0.001
                    
                # We want dist >= 2r. 
                # Current r estimate?
                # Let's use a target radius r_target that increases?
                # Or just maximize the minimum distance.
                # Force = 1/dist^2 is common.
                
                f_mag = 1.0 / (dist * dist)
                fx = f_mag * dx
                fy = f_mag * dy
                
                forces[i, 0] += fx
                forces[i, 1] += fy
                forces[j, 0] -= fx
                forces[j, 1] -= fy
        
        # Update centers
        centers += learning_rate * forces
        
        # Clip to [0, 1] strictly to avoid flying out
        centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
        
        # Cool down learning rate
        if it > 500:
            learning_rate *= 0.999
            
    # 3. Calculate the maximum possible equal radius
    min_dist = 1.0
    for i in range(n):
        x, y = centers[i]
        d_bound = min(x, 1-x, y, 1-y)
        if d_bound < min_dist:
            min_dist = d_bound
            
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d_circ = np.sqrt(dx**2 + dy**2)
            if d_circ < min_dist:
                min_dist = d_circ
                
    r = min_dist / 2.0
    
    # 4. Final adjustment to ensure validity and maximize r
    # The optimization might have drifted slightly.
    # We scale radii to be exactly r.
    radii = np.full(n, r)
    
    # Verify and if invalid, shrink r slightly
    # The force method tries to maximize min_dist, so r should be valid.
    # But numerical errors might occur.
    
    # Let's perform a quick validation loop to shrink r if needed
    valid = False
    while not valid:
        valid = True
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                r *= 0.999
                valid = False
                break
        if not valid: continue
        
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < radii[i] + radii[j] - 1e-9:
                    r *= 0.999
                    radii = np.full(n, r)
                    valid = False
                    break
            if not valid: break
                
        radii = np.full(n, r)

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Helper to run and print result for verification during development
if __name__ == "__main__":
    import numpy as np
    
    # We need to import validate_packing if we want to test, 
    # but the prompt says we can't modify it, just use it.
    # We will assume the logic holds.
    
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Radius: {r[0]}")
