# sol_000350 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1c8d6f7d) state=1d4cc69b sum of radii=2.387207 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed layout with growing radii and vectorized operations.
    """
    np.random.seed(42)
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # Generate a hexagonal lattice and pick the n points closest to the center.
    r_init = 0.05
    dx = 2 * r_init
    dy = math.sqrt(3) * r_init
    
    points = []
    # Generate a grid large enough to contain 26 points
    for row in range(10):
        for col in range(10):
            x = col * dx + (row % 2) * (dx / 2)
            y = row * dy
            points.append([x, y])
    
    points = np.array(points)
    # Center the grid at (0.5, 0.5)
    centroid = np.mean(points, axis=0)
    points = points - centroid + np.array([0.5, 0.5])
    
    # Sort by distance to center (0.5, 0.5) and pick n closest
    center_point = np.array([0.5, 0.5])
    dists = np.linalg.norm(points - center_point, axis=1)
    indices = np.argsort(dists)[:n]
    centers = points[indices].copy()
    
    radii = np.ones(n) * r_init
    
    # 2. Optimization
    # We will iteratively increase the radius R and resolve overlaps using forces.
    
    R = r_init
    dR = 0.0002 # Radius growth step
    max_iterations = 2000
    step_size = 0.005
    
    for it in range(max_iterations):
        radii[:] = R
        
        # Optimization sub-loop for fixed R
        # Run for a few steps to settle positions
        sub_steps = 15
        current_step = step_size / (1.0 + it * 0.001)
        
        for _ in range(sub_steps):
            forces = np.zeros((n, 2))
            
            # Boundary forces
            # Push circles away from boundaries if they penetrate
            for i in range(n):
                x, y = centers[i]
                # Left boundary
                if x < R:
                    forces[i, 0] += (R - x) * 100.0
                # Right boundary
                if x > 1 - R:
                    forces[i, 0] -= (x - (1 - R)) * 100.0
                # Bottom boundary
                if y < R:
                    forces[i, 1] += (R - y) * 100.0
                # Top boundary
                if y > 1 - R:
                    forces[i, 1] -= (y - (1 - R)) * 100.0
            
            # Vectorized pairwise repulsion
            # Compute all pairwise vectors and distances
            diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diffs**2, axis=2)
            dists = np.sqrt(dist_sq)
            
            # Set diagonal to infinity to avoid self-interaction
            np.fill_diagonal(dists, np.inf)
            
            # Identify overlapping pairs (distance < 2*R)
            overlap_mask = dists < 2 * R
            
            # Calculate repulsive force magnitude
            # Force proportional to overlap depth
            mag = np.zeros((n, n))
            mag[overlap_mask] = (2 * R - dists[overlap_mask]) * 200.0
            
            # Calculate force direction (unit vector from j to i)
            # diffs[i, j] is vector from j to i
            with np.errstate(divide='ignore', invalid='ignore'):
                dirs = diffs / dists[:, :, np.newaxis]
            dirs = np.nan_to_num(dirs)
            
            # Sum forces for each circle
            # forces[i] += sum_j (mag[i, j] * dirs[i, j])
            forces_update = np.sum(mag[:, :, np.newaxis] * dirs, axis=1)
            forces += forces_update
            
            # Update centers
            centers += current_step * forces
            centers = np.clip(centers, 0, 1)
        
        # Check validity to decide whether to increase radius
        max_overlap = 0.0
        valid_boundary = True
        
        # Check boundary validity
        for i in range(n):
            x, y = centers[i]
            if x < R or x > 1 - R or y < R or y > 1 - R:
                valid_boundary = False
                ov = 0
                if x < R: ov = max(ov, R - x)
                if x > 1 - R: ov = max(ov, x - (1 - R))
                if y < R: ov = max(ov, R - y)
                if y > 1 - R: ov = max(ov, y - (1 - R))
                max_overlap = max(max_overlap, ov)
        
        if valid_boundary:
            # Check circle-circle overlaps
            # Recompute distances for accuracy
            diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diffs**2, axis=2)
            dists = np.sqrt(dist_sq)
            np.fill_diagonal(dists, np.inf)
            
            # Check minimum distance
            min_dist = np.min(dists[np.triu_indices(n, k=1)])
            if min_dist < 2 * R:
                max_overlap = max(max_overlap, 2 * R - min_dist)
        
        # Adjust radius based on overlap
        if max_overlap < 1e-6:
            # Valid configuration, try to increase radius
            R += dR
        else:
            # Overlap detected, reduce radius to recover validity
            R -= max_overlap / 2.0
            if R < r_init:
                R = r_init

    # Final adjustment to ensure strict validity with tolerance
    max_ov = 0.0
    
    # Check overlaps
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diffs**2, axis=2)
    dists = np.sqrt(dist_sq)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists[np.triu_indices(n, k=1)])
    if min_dist < 2 * R:
        max_ov = max(max_ov, 2 * R - min_dist)
        
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        if x < R: max_ov = max(max_ov, R - x)
        if x > 1 - R: max_ov = max(max_ov, x - (1 - R))
        if y < R: max_ov = max(max_ov, R - y)
        if y > 1 - R: max_ov = max(max_ov, y - (1 - R))
    
    # Shrink radius to satisfy constraints
    final_R = R - max_ov - 1e-9
    if final_R < 0: final_R = 0
    radii[:] = final_R
    
    return centers, radii, np.sum(radii)
