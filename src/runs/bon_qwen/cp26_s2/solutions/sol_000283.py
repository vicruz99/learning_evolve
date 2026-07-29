# sol_000283 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b0b06613) state=cc1ab596 sum of radii=2.195198 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses a force-directed layout with iterative radius expansion.
    """
    n = 26
    
    # 1. Initialization
    # Generate 26 points distributed roughly evenly in the square.
    # Using a grid sorted by Hilbert-like order (z-order) for better distribution than pure random.
    grid_side = 6  # 6x6 = 36 points, we take 26
    indices = np.arange(grid_side * grid_side)
    
    # Z-order (Morton code) sorting for better space-filling
    # Simple approximation: interleave bits, or just sort by row then column?
    # A simple snake pattern or just taking first 26 from a dense grid is fine if we relax.
    # Let's use a simple grid but skip some to get 26, or just take 26 from a denser grid.
    
    # Let's create a 6x6 grid of points
    x = np.linspace(0.1, 0.9, grid_side)
    y = np.linspace(0.1, 0.9, grid_side)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.flatten(), yy.flatten()]).T
    
    # Shuffle or reorder to ensure we pick a good subset?
    # Actually, taking the first 26 from a well-distributed set is fine.
    # Let's just take the first 26.
    # To make it more "Hilbert" like, we can interleave indices.
    idx = np.argsort(indices) # Just identity, but let's mix it up slightly
    # Actually, standard row-major is fine.
    
    # Let's try to fit 26 points in a hexagonal pattern for better initial density.
    # But grid is easier to code. Let's stick to grid but maybe add a small random jitter.
    centers = points[:n].copy()
    # Add tiny jitter to break symmetry
    centers += np.random.uniform(-0.001, 0.001, size=centers.shape)

    # 2. Optimization Loop
    # We will increase radius r and resolve collisions.
    r = 0.01
    max_r_target = 0.102 # Slightly above estimated optimal 0.1014
    
    # Parameters for relaxation
    alpha = 0.5 # Step size for position update
    damping = 0.95 # Damping factor for velocity/forces? We use simple Euler, so alpha acts as step.
    
    # We will perform many small steps of radius increase
    num_steps = 2000
    dr = (max_r_target - 0.01) / num_steps
    
    for step in range(num_steps):
        r += dr
        
        # Relaxation iterations
        # More iterations when r is small, fewer when large?
        # Constant number is safe.
        num_relax = 50
        
        for _ in range(num_relax):
            forces = np.zeros_like(centers)
            
            # 1. Pairwise Repulsion
            # Vectorized calculation for O(N^2)
            # dist_matrix[i, j] = distance between i and j
            # We can compute differences
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # Shape (n, n, 2)
            dists = np.sqrt(np.sum(diff**2, axis=2)) # Shape (n, n)
            
            # Create mask for overlaps (dist < 2r) and i != j
            # Avoid self-interaction
            np.fill_diagonal(dists, np.inf) 
            
            # Overlap amount
            overlap = 2 * r - dists
            # Only consider positive overlaps
            is_overlapping = overlap > 0
            
            # Force magnitude proportional to overlap
            # Direction is diff / dist
            # Be careful with dist=0
            safe_dists = np.where(dists < 1e-9, 1e-9, dists)
            
            # Unit vectors
            # u = diff / safe_dists # Shape (n, n, 2)
            # To save memory/time, we can compute forces directly
            
            # For each i, sum forces from j where overlap > 0
            # F_i += sum_j ( (diff[i,j] / dist[i,j]) * overlap[i,j] )
            
            # Let's do this loop-wise for clarity and to avoid large intermediate arrays if N was big, 
            # but for N=26, vectorized is fine.
            
            # Compute forces matrix
            # Forces on i from j
            # F_ij = diff[i,j] * (overlap[i,j] / dist[i,j])
            # Note: overlap is scalar, dist is scalar.
            
            # Create a mask to ignore non-overlaps and self
            mask = is_overlapping & (dists < 10) # dist < 10 is always true
            
            # Calculate force vectors
            # diff is (n, n, 2)
            # We want to multiply diff by (overlap / dist) scaled by mask
            # overlap/dist is (n, n)
            
            factor = np.where(mask, overlap / safe_dists, 0.0) # (n, n)
            
            # Force contribution from j to i is factor[i,j] * diff[i,j]
            # But diff[i,j] = center[i] - center[j]
            # This pushes i away from j.
            # Sum over j.
            
            # Expand factor to (n, n, 1)
            factor_3d = factor[:, :, np.newaxis]
            forces_vec = diff * factor_3d # (n, n, 2)
            
            # Sum over axis 1 (j)
            forces += np.sum(forces_vec, axis=1)
            
            # 2. Boundary Repulsion
            # If x < r, force right. If x > 1-r, force left.
            # Strength should be comparable to pairwise forces.
            # Pairwise force scale ~ overlap.
            
            # Left boundary
            left_penetration = r - centers[:, 0]
            mask_left = left_penetration > 0
            forces[mask_left, 0] += left_penetration[mask_left] * 5.0 # Stronger boundary force to prevent sticking
            
            # Right boundary
            right_penetration = centers[:, 0] - (1.0 - r)
            mask_right = right_penetration > 0
            forces[mask_right, 0] -= right_penetration[mask_right] * 5.0
            
            # Bottom boundary
            bottom_penetration = r - centers[:, 1]
            mask_bottom = bottom_penetration > 0
            forces[mask_bottom, 1] += bottom_penetration[mask_bottom] * 5.0
            
            # Top boundary
            top_penetration = centers[:, 1] - (1.0 - r)
            mask_top = top_penetration > 0
            forces[mask_top, 1] -= top_penetration[mask_top] * 5.0
            
            # Update positions
            centers += alpha * forces
            
            # Clamp to valid region strictly
            centers[:, 0] = np.clip(centers[:, 0], r, 1.0 - r)
            centers[:, 1] = np.clip(centers[:, 1], r, 1.0 - r)
            
            # Check if system is stable? 
            # Not strictly necessary, just iterate.
            
        # Optional: If r gets too large and we can't resolve, maybe decrease step?
        # But for this problem, 0.1014 is reachable.

    # 3. Finalize
    # The radius used in the last step might be slightly optimistic if overlaps remain.
    # Let's calculate the actual minimum separation / boundary distance to determine valid r.
    # But usually, the clamping and repulsion handles it.
    # To be safe, we can check overlaps and reduce r slightly if needed.
    
    # Check overlaps
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_dist = np.min(dists)
    
    # Current r might be slightly too large if min_dist < 2r
    # We want r such that 2r <= min_dist (and boundary constraints).
    # Also check boundaries
    min_x = np.min(centers[:, 0])
    max_x = np.max(centers[:, 0])
    min_y = np.min(centers[:, 1])
    max_y = np.max(centers[:, 1])
    
    r_boundary_x = min(min_x, 1.0 - max_x)
    r_boundary_y = min(min_y, 1.0 - max_y)
    r_boundary = min(r_boundary_x, r_boundary_y)
    
    r_geom = min_dist / 2.0
    r_bound = r_boundary
    
    # The valid radius is limited by geometry and boundaries
    # However, we expanded r up to ~0.102. 
    # If min_dist is slightly less than 2r, we just take min_dist/2.
    # But we want to report the radii of the circles we packed.
    # If we report r = min(r_geom, r_bound), it's valid.
    
    final_r = min(r_geom, r_bound)
    
    # To be safe against numerical errors, shrink slightly
    final_r = final_r * 0.99999
    
    radii = np.full(n, final_r)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Run the function to ensure it works locally if this were a script
# But we just need to define it.
