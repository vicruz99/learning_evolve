# sol_000033 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f550adc) state=1111046a sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed optimization approach starting from a hexagonal grid.
    """
    n_circles = 26
    
    # Helper function to calculate distances and forces
    # Returns forces to push circles apart and towards valid region
    def compute_forces(centers, r, repulsion_strength=100.0, wall_strength=50.0):
        forces = np.zeros_like(centers)
        
        # 1. Repulsion between circles
        # Vectorized distance calculation
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero and self-interaction
        np.fill_diagonal(dists, np.inf)
        
        # We only care about pairs that are closer than 2*r
        # To save computation and avoid singularity, we compute forces for close pairs
        # However, for dense packing, many pairs are close.
        # Let's use a soft repulsion that is strong when dist < 2r
        
        # Mask for pairs where dist < 2r
        mask = dists < 2.0 * r
        
        # Calculate direction vectors (unit vectors)
        # Normalize diff. Handle zero distances carefully (though mask handles dist < 2r)
        # To avoid div by 0, add small epsilon to dists before normalization if needed,
        # but here dists > 0 for i!=j usually.
        
        # Force magnitude: F = k * (2r - dist)
        # We want to push apart. Force on i from j is along (p_i - p_j)
        
        # Vectorized update
        # We can compute forces only for masked pairs to be efficient, 
        # but with N=26, N^2 = 676 is small.
        
        # Direction
        # dists is (N, N). diff is (N, N, 2).
        # Normalize diff along last axis
        # Add epsilon to avoid division by zero
        dists_safe = np.where(dists < 1e-9, 1e-9, dists)
        unit_diff = diff / dists_safe[:, :, np.newaxis]
        
        # Force magnitude
        # Only apply if dist < 2r
        force_magnitude = np.maximum(0, 2.0 * r - dists)
        force_magnitude *= mask # Zero out if not overlapping/too close
        
        # Accumulate forces
        # force_magnitude is scalar per pair. Multiply by unit_diff to get vector.
        # Sum over j for each i.
        # forces[i] += sum_j (force_magnitude[i,j] * unit_diff[i,j])
        
        # Broadcasting: force_magnitude (N, N, 1) * unit_diff (N, N, 2)
        pair_forces = force_magnitude[:, :, np.newaxis] * unit_diff
        
        # Sum over axis 1 (neighbors)
        net_repulsion = np.sum(pair_forces, axis=1)
        
        forces += net_repulsion * repulsion_strength
        
        # 2. Wall repulsion
        # Walls at x=0, x=1, y=0, y=1
        # If center x < r, push right. Force proportional to (r - x)
        # x coordinates: centers[:, 0]
        # y coordinates: centers[:, 1]
        
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Left wall
        dist_left = x
        overlap_left = np.maximum(0, r - dist_left)
        forces[:, 0] += overlap_left * wall_strength
        
        # Right wall
        dist_right = 1.0 - x
        overlap_right = np.maximum(0, r - dist_right)
        forces[:, 0] -= overlap_right * wall_strength
        
        # Bottom wall
        dist_bottom = y
        overlap_bottom = np.maximum(0, r - dist_bottom)
        forces[:, 1] += overlap_bottom * wall_strength
        
        # Top wall
        dist_top = 1.0 - y
        overlap_top = np.maximum(0, r - dist_top)
        forces[:, 1] -= overlap_top * wall_strength
        
        return forces

    # Initial placement: Hexagonal grid approximation
    # We want to pack 26 circles.
    # A 5x5 grid has 25. We can try to perturb it or use a hexagonal pattern.
    # Hexagonal pattern rows: 5, 5, 5, 5, 6? Or 6, 5, 5, 5, 5?
    # Let's try to arrange them in a way that fits well.
    # Maybe 3 rows of 9? No.
    # Let's generate points in a hexagonal lattice and select 26 that fit in [0,1].
    
    # Estimate radius r_init ~ 0.1
    r_init = 0.1
    
    # Hexagonal spacing
    # Horizontal spacing dx = 2*r
    # Vertical spacing dy = sqrt(3)*r
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    
    points = []
    
    # Generate a grid of points
    # x from 0 to 1, y from 0 to 1
    # Row 0: x = 0, 2r, 4r...
    # Row 1: x = r, 3r, 5r... (offset by r)
    
    y = 0.0
    row_idx = 0
    while y <= 1.0:
        offset = 0.0 if row_idx % 2 == 0 else r_init
        x = offset
        while x <= 1.0:
            points.append([x, y])
            x += dx
        y += dy
        row_idx += 1
    
    points = np.array(points)
    
    # Filter points that are strictly inside [0,1] with some margin?
    # Actually, we just need 26 points.
    # If we have more than 26, we might need to select the best subset?
    # Or just take the first 26.
    # The lattice generation might produce points outside [0,1] if we don't check.
    # But the loop condition x<=1.0 and y<=1.0 ensures centers are in [0,1].
    # However, circles of radius r_init might stick out.
    # We need centers in [r, 1-r].
    # Let's adjust generation to be safe.
    
    # Better initialization: Random perturbation of a grid
    # Or just a dense random start?
    # Let's use the grid points but ensure they are feasible for a smaller r.
    
    # Let's just take the generated points and clip/keep first 26.
    if len(points) < 26:
        # Fallback to random if not enough (unlikely with r=0.1)
        points = np.random.rand(26, 2)
    else:
        # Take first 26
        points = points[:26]
        
    centers = points.astype(float)
    
    # Optimization loop
    # We will try to increase r.
    # In each step, we run the force simulation to relax the configuration.
    
    r = r_init
    max_iter = 500
    steps_per_r = 100
    
    # We can perform a binary search or incremental search for optimal r.
    # Let's do incremental increase.
    
    r_step = 0.0005
    
    # To make it robust, we can run the optimizer for a fixed number of steps
    # increasing r slowly.
    
    current_r = 0.05 # Start small to ensure feasibility
    
    # Initial centers spread out
    # Use a grid to start
    centers = np.zeros((26, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < 26:
                centers[idx] = [0.1 + j * 0.2, 0.1 + i * 0.2] # 0.1 offset, 0.2 spacing
                idx += 1
    # 26th circle
    if idx < 26:
        # Add one somewhere, maybe center?
        # Or just random
        centers[idx] = [0.5, 0.5] 
        # Overlap will be resolved by forces
    
    # Refine initial centers to be a hexagonal packing
    # Let's create a proper hexagonal packing for 26 circles.
    # Rows: 6, 5, 5, 5, 5 is 26.
    # But width constraint.
    # Let's just use the force simulation to find a good spot.
    
    # Reset centers to a dense random packing or specific pattern
    # Pattern: 5 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # ...
    # Total 25. Add 1.
    
    # Let's try a 5x5 grid + 1 center
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([j * 0.2 + 0.1, i * 0.2 + 0.1])
    centers.append([0.5, 0.5]) # 26th
    centers = np.array(centers)
    
    # Shuffle to avoid symmetry bias?
    # np.random.shuffle(centers)
    
    # Optimization
    # We want to find max r.
    # We can run a loop: try r, relax, if relaxed energy > threshold, fail.
    
    best_r = 0.0
    best_centers = centers.copy()
    
    # Adaptive step size
    dr = 0.001
    
    # We can just run a long optimization where r increases slowly.
    # But finding the exact max is tricky.
    # Let's run a fixed optimization with increasing r target.
    
    target_r = 0.08
    iterations = 2000
    
    learning_rate = 0.01
    
    for step in range(iterations):
        # Increase target r slowly
        if step % 50 == 0:
            target_r += 0.0005
        
        # Compute forces based on current target_r?
        # No, forces should push to satisfy constraints for current r.
        # But if we want to maximize r, we should compute forces for the current configuration's effective r?
        # Or just use a fixed large repulsion.
        
        # Actually, a common trick is to treat r as a variable or just check overlap.
        # Let's compute forces with a 'repulsion radius' slightly larger than current circle size to push them apart.
        # But we don't have a fixed radius yet.
        
        # Alternative: Just minimize the "overlap energy" with a fixed radius guess, then increase guess.
        # But let's stick to the force method with a dynamic radius.
        
        # Let's assume current radius is 'r'.
        # If circles overlap, push apart.
        # If they are not overlapping, we can try to increase r.
        
        # But we need to track r.
        # Let's assume all circles have radius r.
        # We start with small r.
        
        # Check overlaps with current r
        # If overlaps exist, push apart.
        # If no overlaps, increase r slightly.
        
        # However, simply increasing r might cause overlaps immediately.
        # So we need to maintain a balance.
        
        # Let's define a 'pressure' or 'target separation'.
        # We want separation >= 2*r.
        
        # Let's use the computed forces function.
        forces = compute_forces(centers, target_r, repulsion_strength=50.0, wall_strength=50.0)
        
        # Update centers
        centers += learning_rate * forces
        
        # Clamp centers to [0, 1]
        centers = np.clip(centers, 0, 1)
        
        # Check if configuration is valid for target_r?
        # The forces do that.
        
        # Adaptive learning rate decay
        if step > 1000:
            learning_rate *= 0.99
            
        # Increase target_r if things look stable?
        # Hard to tell.
        # Let's just increase target_r continuously.
        
        # Check validity periodically
        if step % 100 == 0:
            # Verify if target_r is feasible
            # Compute min distance
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            min_wall_dist = np.min(np.array([
                centers[:, 0], 1 - centers[:, 0],
                centers[:, 1], 1 - centers[:, 1]
            ]))
            
            feasible_r = min(min_dist / 2.0, min_wall_dist)
            
            if feasible_r > target_r:
                # We can increase target_r faster
                target_r += 0.001
            else:
                # Maybe decrease or stay
                # If we are stuck, reduce target_r to escape local min?
                # Or just keep trying.
                pass
    
    # Final verification and cleanup
    # Determine the actual feasible radius for the final centers
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    min_wall_dist = np.min(np.array([
        centers[:, 0], 1 - centers[:, 0],
        centers[:, 1], 1 - centers[:, 1]
    ]))
    
    r_final = min(min_dist / 2.0, min_wall_dist)
    
    # To be safe, reduce r_final slightly to avoid numerical errors
    r_final = r_final * 0.9999
    
    # Check if we have 26 circles
    # If the optimization failed and collapsed, we might have issues.
    # But with repulsion, they should spread.
    
    # Construct radii array
    radii = np.full(26, r_final)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
