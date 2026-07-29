# sol_000022 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f223c9a2) state=49ac302f sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal initialization followed by a force-directed expansion.
    """
    n = 26
    
    # --- 1. Initialization: Hexagonal Grid Pattern ---
    # We aim to place points in a hexagonal lattice.
    # A 5x5 grid gives 25 points. We add one more or adjust.
    # Let's try to fit rows with counts like 6, 5, 6, 5, 4 (Sum = 26)
    # or 5, 6, 5, 6, 4 (Sum = 26).
    # Hexagonal spacing: horizontal 2r, vertical sqrt(3)r, shift r.
    # Since we don't know r, we use a unit grid and let the optimizer scale it.
    
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Initialize centers in a hexagonal pattern
    # We will generate points and then normalize/scale them if necessary,
    # but placing them directly in [0,1] with some spacing is better.
    
    # Let's create a pattern of rows.
    # Row 0: 6 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 6 circles
    # Row 3: 5 circles (shifted)
    # Row 4: 4 circles
    # Total: 26
    
    row_counts = [6, 5, 6, 5, 4]
    current_idx = 0
    
    # Estimated vertical spacing to fit 5 rows in [0, 1]
    # We want circles to be somewhat separated initially.
    # Let's place them with a fixed spacing.
    # Horizontal spacing dx = 0.18, Vertical spacing dy = 0.18 * sqrt(3) approx 0.31?
    # Actually, let's just use a grid spacing of 0.2 and adjust.
    
    # Better: Just distribute them roughly evenly first, then let forces work.
    # But a hex grid is better.
    
    # Let's set a scale factor. If we want 6 circles in width 1, spacing ~ 1/6 = 0.166.
    # Let's use spacing s = 0.18.
    s = 0.18
    dy = s * math.sqrt(3)
    
    # Center vertically in the square.
    total_height = (len(row_counts) - 1) * dy
    y_start = (1 - total_height) / 2 + s # Add s to account for radius later? No, just center.
    # Actually, let's just start at y=0.1 and go up.
    y_pos = 0.15 
    
    for r_idx, count in enumerate(row_counts):
        # Shift for hexagonal packing
        if r_idx % 2 == 1:
            x_start = 0.15 + s/2 # Shifted row
        else:
            x_start = 0.15
        
        for c_idx in range(count):
            if current_idx < n:
                x = x_start + c_idx * s
                # Clamp to [0, 1] roughly
                x = max(0.0, min(1.0, x))
                y = y_pos
                
                centers[current_idx, 0] = x
                centers[current_idx, 1] = y
                current_idx += 1
        
        y_pos += dy

    # If we didn't place all (should be 26), fill remaining randomly
    if current_idx < n:
        for i in range(current_idx, n):
            centers[i, 0] = np.random.uniform(0, 1)
            centers[i, 1] = np.random.uniform(0, 1)
            current_idx = i + 1

    # Initial small radii to allow movement
    radii[:] = 0.01

    # --- 2. Physics Simulation (Force-Directed Layout) ---
    
    dt = 0.01
    velocities = np.zeros_like(centers)
    damping = 0.9
    repulsion_strength = 200.0 # Stiffness of repulsion
    expansion_rate = 0.0002 # How fast radii grow
    
    # Run simulation for a fixed number of steps
    # We can run more steps if needed, but time limit is a concern.
    # 2000 steps should be enough for 26 circles.
    steps = 3000
    
    # Precompute indices for pairs to avoid loops in Python if possible, 
    # but N=26 is small enough for O(N^2) loop.
    
    for step in range(steps):
        # Expand radii
        # We can expand all radii uniformly
        radii += expansion_rate
        
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # Using vectorized operations for speed
        # Centers shape (N, 2)
        # Compute differences: (i, j, 2)
        # This creates an N x N x 2 array
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-9) # Avoid div by zero
        dist = np.sqrt(dist_sq)
        
        # Radii sum matrix (N, N)
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap: positive if dist < r_sum
        # We want force proportional to (r_sum - dist)
        overlap = r_sum - dist
        
        # Only apply force if overlap > 0
        mask = overlap > 0
        
        # Normalize diff vector
        # diff / dist. If dist is 0, diff is 0, so direction is 0. Safe.
        norm_diff = diff / dist[:, :, np.newaxis]
        
        # Force magnitude = overlap * strength
        # But we need to be careful not to explode forces.
        force_mag = overlap * repulsion_strength
        
        # Apply force to the force array
        # This is a bit tricky to vectorize fully for accumulation without loops or broadcasting tricks
        # Loop over pairs is fine for N=26
        
        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j]:
                    # Force vector on i
                    f_vec = norm_diff[i, j] * force_mag[i, j]
                    forces[i] += f_vec
                    forces[j] -= f_vec
        
        # Wall repulsion
        # If center - radius < 0, push right. Force ~ (radius - center)
        # If center + radius > 1, push left. Force ~ (1 - (center + radius))
        
        # Left wall
        wall_dist_x_left = centers[:, 0] - radii
        wall_dist_x_right = 1 - (centers[:, 0] + radii)
        wall_dist_y_bottom = centers[:, 1] - radii
        wall_dist_y_top = 1 - (centers[:, 1] + radii)
        
        # Force if distance is negative (penetration)
        # We can use a soft repulsion for small penetrations to prevent explosion
        # But hard repulsion works too. Let's use proportional to penetration.
        
        # X forces
        mask_x_left = wall_dist_x_left < 0
        forces[mask_x_left, 0] += -wall_dist_x_left[mask_x_left] * repulsion_strength
        
        mask_x_right = wall_dist_x_right < 0
        forces[mask_x_right, 0] += wall_dist_x_right[mask_x_right] * repulsion_strength # Note: dist is negative, so adding negative pushes left?
        # Wait, if wall_dist_x_right < 0, it means center + r > 1.
        # We want to push left (negative force).
        # wall_dist_x_right is negative. So adding it (negative) is correct?
        # Let's check: center=0.9, r=0.2. center+r=1.1. dist = 1 - 1.1 = -0.1.
        # Force += -0.1 * K. Force is negative. Pushes left. Correct.
        
        # Y forces
        mask_y_bottom = wall_dist_y_bottom < 0
        forces[mask_y_bottom, 1] += -wall_dist_y_bottom[mask_y_bottom] * repulsion_strength
        
        mask_y_top = wall_dist_y_top < 0
        forces[mask_y_top, 1] += wall_dist_y_top[mask_y_top] * repulsion_strength

        # Update velocities
        velocities += forces * dt
        velocities *= damping
        
        # Update positions
        centers += velocities * dt
        
        # Clamp positions strictly to valid range to prevent drift out
        # Valid range for center is [r, 1-r]
        # But since r changes, we just clamp to [0, 1] effectively, 
        # relying on wall forces to keep them in [r, 1-r].
        # However, if they bounce out, we clip.
        # A safer clamp:
        min_c = radii
        max_c = 1 - radii
        
        # Clip X
        centers[:, 0] = np.clip(centers[:, 0], min_c, max_c)
        # Clip Y
        centers[:, 1] = np.clip(centers[:, 1], min_c, max_c)

    # --- 3. Final Adjustment ---
    # The simulation might leave small overlaps due to discrete time steps.
    # We can do a few correction steps or just rely on the fact that forces were strong.
    # Let's reduce radii slightly if overlaps exist to ensure validity for the checker.
    # Or, since the checker allows 1e-12 error, it should be fine.
    # But to be safe, let's verify and shrink if needed.
    
    # Check for overlaps
    # If overlaps exist, shrink radii uniformly until valid.
    # This is a conservative step but ensures validity.
    
    # Re-validate
    valid = False
    while not valid:
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-9:
                    # Overlap detected. Shrink radii.
                    # Shrink all radii by a small amount to resolve.
                    radii -= 0.0001
                    valid = False
                    break
            if not valid: break
        
        # Also check boundary
        if valid:
            for i in range(n):
                if (centers[i, 0] < radii[i] - 1e-9 or 
                    centers[i, 0] > 1 - radii[i] + 1e-9 or
                    centers[i, 1] < radii[i] - 1e-9 or
                    centers[i, 1] > 1 - radii[i] + 1e-9):
                    radii[i] -= 0.0001
                    valid = False
                    break
                    
    # Ensure non-negative radii
    radii = np.maximum(radii, 0.0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
