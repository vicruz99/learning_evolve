import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # --- Configuration ---
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # 1. Initialization: Hybrid Hexagonal + Perturbation
    # We create a mix of large and small circles to allow for high density.
    
    # Place 5 large circles in a cross/pentagon pattern
    r_large = 0.15
    # Center and 4 corners
    large_indices = [0, 1, 2, 3, 4]
    centers[0] = [0.5, 0.5]
    centers[1] = [0.25, 0.25]
    centers[2] = [0.75, 0.25]
    centers[3] = [0.25, 0.75]
    centers[4] = [0.75, 0.75]
    radii[:5] = r_large
    
    # Place 21 smaller circles in the gaps (perturbed grid)
    r_small = 0.06
    idx = 5
    # Generate a dense grid and pick points far from existing circles
    # Simple grid approach for the rest
    grid_step = 0.2
    for x in np.arange(0.1, 1.0, grid_step):
        for y in np.arange(0.1, 1.0, grid_step):
            if idx >= n:
                break
            # Check distance to large circles
            is_valid = True
            for k in range(5):
                dist = np.sqrt((x - centers[k][0])**2 + (y - centers[k][1])**2)
                if dist < r_large + r_small + 0.05: # loose check
                    is_valid = False
                    break
            
            if is_valid:
                centers[idx] = [x, y]
                radii[idx] = r_small
                idx += 1
    
    # Fill remaining if needed
    while idx < n:
        centers[idx] = [np.random.uniform(0, 1), np.random.uniform(0, 1)]
        radii[idx] = r_small
        idx += 1

    # Add small noise to break symmetry
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    radii += np.random.uniform(-0.001, 0.001, size=radii.shape)
    radii = np.maximum(radii, 0.001)

    # 2. Optimization: Gradient Descent / Repulsive Forces
    # Objective: Maximize sum(radii) by growing them and resolving overlaps
    
    learning_rate = 0.5
    growth_rate = 0.0002
    steps = 3000
    
    # Precompute indices for O(N^2) loop efficiency inside
    n_idx = np.arange(n)
    
    for step in range(steps):
        # Grow radii slightly
        radii += growth_rate
        
        # Compute forces (repulsion and boundary)
        forces = np.zeros_like(centers)
        
        # 1. Circle-Circle Repulsion
        # Vectorized pair-wise distance calculation
        # Shape: (n, n, 2) differences
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Mask lower triangle to avoid double counting and self-interaction
        # dists is (n, n). Mask where i < j
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        
        # Penetration depth: max(0, r_i + r_j - dist)
        sum_radii_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
        penetration = np.maximum(0, sum_radii_matrix - dists)
        
        # Force magnitude proportional to penetration
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        
        # Unit vector direction (force pushes i away from j)
        # For i<j, force on i is along (c_i - c_j)
        # We accumulate into forces array
        unit_vecs = np.where(diffs != 0, diffs / safe_dists[:, :, np.newaxis], 0.0)
        
        # Apply force: F_ij = penetration_ij * unit_vec_ij
        # Force on i += penetration * dir
        # Force on j -= penetration * dir
        force_magnitude = penetration * mask
        
        # Accumulate forces
        # Sum over j axis
        forces += np.sum(force_magnitude[:, :, np.newaxis] * unit_vecs, axis=1)
        # For the j component (opposite direction)
        forces -= np.sum((force_magnitude.T)[:, :, np.newaxis] * unit_vecs.T, axis=1)

        # 2. Boundary Forces
        # Push circles back inside if they overlap with walls
        # Left wall
        overlap_x_min = np.maximum(0, radii - centers[:, 0])
        forces[:, 0] += overlap_x_min
        
        # Right wall
        overlap_x_max = np.maximum(0, centers[:, 0] - (1 - radii))
        forces[:, 0] -= overlap_x_max
        
        # Bottom wall
        overlap_y_min = np.maximum(0, radii - centers[:, 1])
        forces[:, 1] += overlap_y_min
        
        # Top wall
        overlap_y_max = np.maximum(0, centers[:, 1] - (1 - radii))
        forces[:, 1] -= overlap_y_max

        # Apply forces (Gradient Ascent on positions to reduce overlap)
        # We move centers in the direction of the force to reduce energy
        centers += learning_rate * forces
        
        # Clamp centers to [0, 1] to prevent explosion
        centers = np.clip(centers, 0, 1)

    # 3. Post-processing
    # The radii might be slightly too large due to the growth loop ending.
    # We need to shrink them until valid.
    
    # Compute current overlaps
    def get_max_overlap(c, r):
        d = np.linalg.norm(c[:, np.newaxis, :] - c[np.newaxis, :, :], axis=2)
        s_r = r[:, np.newaxis] + r[np.newaxis, :]
        overlaps = np.maximum(0, s_r - d)
        # Boundary overlaps
        b_overlap = np.maximum(0, np.maximum(r - c[:, 0], c[:, 0] - (1 - r)))
        b_overlap += np.maximum(0, np.maximum(r - c[:, 1], c[:, 1] - (1 - r)))
        return np.max(overlaps) + np.max(b_overlap)

    max_ov = get_max_overlap(centers, radii)
    
    # Shrink radii uniformly to resolve overlaps
    if max_ov > 1e-9:
        # Linear shrinkage factor
        # Approximate: reducing radius by delta reduces overlap by 2*delta (for pairs)
        # Just use a safety shrink
        radii -= max_ov + 0.001
    
    # Ensure non-negative
    radii = np.maximum(radii, 0.0)
    
    # Final validation check (internal)
    # If still overlapping, force shrink more aggressively
    for _ in range(100):
        ov = get_max_overlap(centers, radii)
        if ov <= 1e-12:
            break
        radii -= ov / 2.0
        radii = np.maximum(radii, 0.0)

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii