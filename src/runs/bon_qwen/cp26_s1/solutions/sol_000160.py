# sol_000160 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 929e2415) state=0e2fa799 sum of radii=1.801371 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed growth algorithm starting from a hexagonal grid.
    """
    n = 26
    np.random.seed(42)  # For reproducibility
    
    # 1. Initialization: Perturbed hexagonal grid
    centers = np.zeros((n, 2))
    idx = 0
    # Grid dimensions to cover the square
    cols, rows = 6, 5
    
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Hexagonal shift for odd rows
            shift = 0.5 if i % 2 == 1 else 0.0
            x = (j + shift + 0.5) / cols
            y = (i + 0.5) / rows
            
            # Scale and shift to leave initial margin
            centers[idx] = [x * 0.8 + 0.1, y * 0.8 + 0.1]
            idx += 1
        if idx >= n:
            break
            
    # Add small random perturbation to break symmetry and help escape local minima
    centers += np.random.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # 2. Simulation Parameters
    r = 0.02
    max_iter = 25000
    
    # Pre-allocate arrays for performance
    diff = np.zeros((n, n, 2))
    dist = np.zeros((n, n))
    forces = np.zeros_like(centers)
    
    for step in range(max_iter):
        # Compute pairwise differences and distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e10) # Ignore self-distances
        
        # Compute repulsion forces between circles
        overlap = 2 * r - dist
        mask = overlap > 0
        
        forces[:] = 0.0 # Reset forces
        if np.any(mask):
            # Avoid division by zero
            safe_dist = np.where(dist > 1e-8, dist, 1e-8)
            force_mag = np.where(mask, overlap / safe_dist, 0.0)
            force_vec = diff * force_mag[:, :, np.newaxis]
            forces += np.sum(force_vec, axis=1)
            
        # Compute wall repulsion forces
        for dim in range(2):
            low = centers[:, dim] < r
            high = centers[:, dim] > 1 - r
            forces[low, dim] += (r - centers[low, dim])
            forces[high, dim] -= (centers[high, dim] - (1 - r))
            
        # Update positions with decaying learning rate
        lr = 0.05 * np.exp(-step * 0.0001)
        centers += lr * forces
        
        # Enforce boundaries strictly
        centers = np.clip(centers, [r, r], [1-r, 1-r])
        
        # Adaptive radius growth
        # Grow faster when system is relatively stable (low forces)
        if np.any(mask):
            max_force = np.max(np.linalg.norm(forces, axis=1))
        else:
            max_force = 0.0
            
        if max_force < 1e-4:
            r *= 1.00008
        else:
            r *= 1.00001

    # 3. Final Validation & Correction
    # Recompute distances to check for any remaining overlaps or boundary violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, 1e10)
    min_pair_dist = np.min(dist)
    
    min_wall_dist = np.min([
        np.min(centers[:, 0]), np.min(1 - centers[:, 0]),
        np.min(centers[:, 1]), np.min(1 - centers[:, 1])
    ])
    
    # Determine the largest valid radius
    r_final = r
    if min_pair_dist < 2 * r_final:
        r_final = min_pair_dist / 2.0
    if min_wall_dist < r_final:
        r_final = min_wall_dist
        
    # Apply a tiny safety margin for floating point comparisons in validation
    r_final *= 0.999999
    
    radii = np.full(n, r_final)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
