# sol_000019 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 04e92922) state=0ee78ff1 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses a force-directed optimization approach starting from a hexagonal configuration.
    """
    n_circles = 26
    side_length = 1.0
    
    # --- 1. Initialization ---
    # We start with a hexagonal packing configuration.
    # A hexagonal lattice is denser than a square grid.
    # We need to select 26 points from a hexagonal grid that fit in the square.
    
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Initial radius to ensure we fit easily. 
    # We will grow this during optimization.
    r_init = 0.06 
    
    # Generate a hexagonal grid of points
    # Spacing in x: 2 * r
    # Spacing in y: sqrt(3) * r
    # Shift odd rows by r in x
    
    # We want to generate enough points to pick 26
    # Approximate size of grid needed: sqrt(26) ~ 5 rows
    rows = 7
    cols = 7
    pts = []
    
    dy = math.sqrt(3) * r_init
    dx = 2 * r_init
    
    # Generate points
    for r in range(rows):
        for c in range(cols):
            y = r * dy + r_init # padding r_init to stay inside
            x = c * dx + r_init
            if r % 2 == 1:
                x += r_init # shift for hexagonal pattern
            
            # Check if point is within bounds [r_init, 1-r_init]
            if x >= r_init and x <= side_length - r_init and \
               y >= r_init and y <= side_length - r_init:
                pts.append([x, y])
    
    # If we don't have enough points, fill with random or fallback
    if len(pts) < n_circles:
        # Fallback to random initialization if grid is too sparse (unlikely with r=0.06)
        for i in range(n_circles):
            centers[i] = [np.random.uniform(r_init, side_length-r_init), 
                          np.random.uniform(r_init, side_length-r_init)]
    else:
        # Select the first n_circles points
        # Ideally we should select the best 26, but taking the first ones from the grid is fine
        # The optimizer will adjust them.
        # To make it more robust, we can try to pick points that are most central or just sequential.
        # Sequential from a dense grid is okay.
        pts = np.array(pts[:n_circles])
        centers = pts

    # Initialize radii
    radii[:] = r_init

    # --- 2. Optimization Loop ---
    # We will try to increase the radius and resolve overlaps.
    
    # Parameters
    max_iter = 3000
    dt = 0.05 # Time step for force application
    stiffness = 10.0 # Force constant for overlaps
    wall_stiffness = 50.0 # Force constant for walls
    radius_growth_rate = 0.0001 # How much to increase radius per iteration
    
    # Convert centers to float64 for precision
    centers = centers.astype(np.float64)
    radii = radii.astype(np.float64)
    
    # Current common radius
    r_current = r_init
    
    for step in range(max_iter):
        # Try to grow radius
        r_current += radius_growth_rate
        radii[:] = r_current
        
        # Resolve overlaps and boundary constraints
        # We perform multiple sub-steps per radius growth to stabilize
        sub_steps = 20
        for _ in range(sub_steps):
            forces = np.zeros_like(centers)
            
            # Calculate pairwise repulsion
            # Vectorized distance calculation
            # dist_matrix[i, j] = ||c_i - c_j||
            # Using broadcasting
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (N, N, 2)
            dists = np.sqrt(np.sum(diff**2, axis=2)) # (N, N)
            
            # Overlap amount: (r_i + r_j) - dist
            # We only care about positive overlaps
            # r_i + r_j is constant 2*r_current
            overlap = 2 * r_current - dists
            
            # Mask for overlaps (and avoid self-interaction i==j)
            # dists is 0 on diagonal, overlap will be 2r > 0. We ignore diagonal.
            np.fill_diagonal(overlap, 0)
            
            # Force direction is along the line connecting centers
            # F_ij pushes i away from j
            # F = stiffness * overlap * (c_i - c_j) / dist
            
            # To avoid division by zero, add small epsilon or mask
            # Only compute for overlapping pairs
            is_overlap = overlap > 1e-10
            
            # Normalize direction vector
            # dists can be 0, so handle carefully
            inv_dists = np.zeros_like(dists)
            mask_valid = dists > 1e-10
            inv_dists[mask_valid] = 1.0 / dists[mask_valid]
            
            # Force magnitude
            force_mag = stiffness * np.maximum(0, overlap)
            
            # Direction unit vectors (diff / dist)
            # We can compute force directly: force_vec = force_mag * (diff / dist)
            # But diff/dist is just direction.
            
            # Accumulate forces
            # For each pair (i, j), force on i is +F, on j is -F
            # F vector = force_mag[i,j] * diff[i,j] * inv_dists[i,j]
            
            # Vectorized update
            # Force on i due to all j
            # F_i = sum_j ( stiffness * max(0, 2r - dist_ij) * (c_i - c_j) / dist_ij )
            
            # Note: diff[i,j] = c_i - c_j
            # So we sum over j
            
            # Compute force components
            # force_tensor[i, j, k] = force_mag[i, j] * diff[i, j, k] * inv_dists[i, j]
            # But force_mag and inv_dists are (N, N). diff is (N, N, 2).
            
            # Expand dimensions for broadcasting
            # force_mag: (N, N, 1)
            # inv_dists: (N, N, 1)
            # diff: (N, N, 2)
            
            f_mag_3d = force_mag[:, :, np.newaxis]
            inv_d_3d = inv_dists[:, :, np.newaxis]
            
            # Pairwise force vectors from j to i (direction i - j)
            # Actually diff[i,j] is c_i - c_j.
            # If overlap > 0, we want to push i away from j.
            # Direction c_i - c_j is correct.
            
            pairwise_forces = f_mag_3d * inv_d_3d * diff
            
            # Sum over j (axis 1) to get total force on i
            forces += np.sum(pairwise_forces, axis=1)
            
            # --- Boundary Forces ---
            # If circle i overlaps with left wall (x < r): push right
            # Overlap amount = r - x
            left_overlap = r_current - centers[:, 0]
            left_overlap = np.maximum(0, left_overlap)
            forces[:, 0] += wall_stiffness * left_overlap
            
            # Right wall (x > 1 - r): push left
            right_overlap = centers[:, 0] - (side_length - r_current)
            right_overlap = np.maximum(0, right_overlap)
            forces[:, 0] -= wall_stiffness * right_overlap
            
            # Bottom wall (y < r): push up
            bottom_overlap = r_current - centers[:, 1]
            bottom_overlap = np.maximum(0, bottom_overlap)
            forces[:, 1] += wall_stiffness * bottom_overlap
            
            # Top wall (y > 1 - r): push down
            top_overlap = centers[:, 1] - (side_length - r_current)
            top_overlap = np.maximum(0, top_overlap)
            forces[:, 1] -= wall_stiffness * top_overlap
            
            # Update positions
            # Clip velocities? No, simple Euler integration.
            # Damping to prevent oscillation
            damping = 0.8
            centers += damping * dt * forces
            
            # Hard clamp to keep inside square during simulation to prevent wild excursions
            # Though wall forces should handle it, hard clamping is safe.
            centers[:, 0] = np.clip(centers[:, 0], 0, side_length)
            centers[:, 1] = np.clip(centers[:, 1], 0, side_length)
            
            # Ensure centers are at least r away from walls? 
            # The forces handle this, but hard constraint:
            centers[:, 0] = np.clip(centers[:, 0], r_current, side_length - r_current)
            centers[:, 1] = np.clip(centers[:, 1], r_current, side_length - r_current)

    # --- 3. Final Validation and Cleanup ---
    # Check for any remaining NaNs or invalid states
    if np.isnan(centers).any() or np.isnan(radii).any():
        # Fallback to a safe grid if optimization failed completely
        print("Optimization produced NaN, falling back to grid.")
        centers = np.zeros((n_circles, 2))
        radii = np.zeros(n_circles)
        # 5x5 grid is safe for 25, need 26. 
        # Fallback: 6x6 grid subset with smaller radius
        step = 1.0 / 6.0
        count = 0
        for i in range(6):
            for j in range(6):
                if count < n_circles:
                    centers[count] = [step * (i + 0.5), step * (j + 0.5)]
                    radii[count] = step * 0.4 # Radius slightly less than half step
                    count += 1
        # Adjust remaining
        while count < n_circles:
             centers[count] = [0.5, 0.5]
             radii[count] = 0.0
             count += 1
    else:
        # Ensure radii are consistent (we optimized a common radius)
        # But the problem allows different radii. 
        # We can just return the common radius we found.
        # However, due to wall clamping, some circles might be slightly compressed?
        # We enforced centers in [r, 1-r], so they are valid w.r.t walls.
        # Overlaps? The forces minimized them, but might not be perfect 0.
        # We can run a final shrink to ensure no overlaps if necessary, 
        # but for the purpose of the problem, a valid packing is required.
        # Let's verify and shrink slightly if overlaps exist.
        
        # Check overlaps
        valid = True
        # Re-check distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        sum_r = np.min(radii) * 2 # Assuming equal radii
        
        if min_dist < sum_r - 1e-9:
            # Overlap detected. Shrink radii until valid.
            # This is a safety measure.
            scale_factor = min_dist / (sum_r) * 0.99
            if scale_factor < 1.0:
                radii *= scale_factor
                # Note: shrinking radii doesn't move centers, so overlaps remain if scale is not enough?
                # Actually if we scale radii down, sum_r decreases.
                # We need sum_r <= min_dist.
                # New r = min_dist / 2.
                radii[:] = min_dist / 2.0
        
        # Re-check boundaries
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            if x < r: centers[i, 0] = r
            if x > 1 - r: centers[i, 0] = 1 - r
            if y < r: centers[i, 1] = r
            if y > 1 - r: centers[i, 1] = 1 - r

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
