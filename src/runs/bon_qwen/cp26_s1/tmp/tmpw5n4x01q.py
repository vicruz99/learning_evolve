import numpy as np
import math

def run_packing() -> tuple:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Uses a force-directed simulation (grow and repel) approach starting from 
    a hexagonal lattice initialization.
    """
    n_circles = 26
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # 1. Initialization: Place circles on a hexagonal lattice
    # This provides a good starting distribution for the optimizer.
    # We aim to distribute 26 points as evenly as possible.
    points = []
    s = 0.18  # Initial spacing between centers
    y = 0.15  # Initial y position
    row_idx = 0
    
    # Generate points in a hexagonal pattern
    while y < 1.0 and len(points) < 26:
        offset = (row_idx % 2) * (s / 2)
        x = 0.15 + offset
        while x < 1.0 and len(points) < 26:
            points.append([x, y])
            x += s
        y += s * math.sqrt(3) / 2
        row_idx += 1
    
    # Fallback if not enough points generated (unlikely with these params)
    while len(points) < 26:
        # Try random placement with some spacing
        p = np.random.uniform(0.05, 0.95, 2)
        # Simple check to avoid immediate massive overlap
        if np.all(np.linalg.norm(np.array(points) - p, axis=1) > 0.12):
            points.append(p)
        else:
            # If can't find good spot, just add it (simulation will fix)
            points.append(p)
            break 
    
    init_points = np.array(points[:26])
    centers = init_points.copy()
    radii = np.full(26, 0.05) # Initial small radius
    
    # 2. Optimization Loop (Force-Directed)
    # We grow radii and push circles apart to find a dense packing.
    dt = 0.001 
    k_repulse = 15.0 
    k_boundary = 15.0 
    growth_rate = 0.0005 
    
    max_iterations = 6000
    
    for step in range(max_iterations):
        # Grow radii uniformly
        radii += growth_rate
        
        forces = np.zeros_like(centers)
        
        # --- Boundary Forces ---
        # Push circles away from boundaries if they intersect
        
        # Left boundary (x < r)
        mask = centers[:, 0] < radii
        forces[mask, 0] += k_boundary * (radii[mask] - centers[mask, 0])
        
        # Right boundary (x > 1-r)
        mask = centers[:, 0] > 1.0 - radii
        forces[mask, 0] -= k_boundary * (centers[mask, 0] - (1.0 - radii[mask]))
        
        # Bottom boundary (y < r)
        mask = centers[:, 1] < radii
        forces[mask, 1] += k_boundary * (radii[mask] - centers[mask, 1])
        
        # Top boundary (y > 1-r)
        mask = centers[:, 1] > 1.0 - radii
        forces[mask, 1] -= k_boundary * (centers[mask, 1] - (1.0 - radii[mask]))
        
        # --- Inter-circle Repulsion Forces ---
        # If circles overlap, push them apart
        
        # Vectorized distance calculation
        # diffs[i, j] = centers[i] - centers[j]
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Ignore self-distances
        np.fill_diagonal(dists, np.inf)
        
        # Sum of radii matrix
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount (positive if dist < r_sum)
        overlap = r_sums - dists
        
        # Only apply force if overlapping
        is_overlapping = overlap > 1e-7
        overlap_safe = np.where(is_overlapping, overlap, 0.0)
        
        # Direction vectors (unit vectors from j to i)
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        unit_diffs = diffs / safe_dists[:, :, np.newaxis]
        
        # Force magnitude proportional to overlap
        # Force on i from j is in direction (c_i - c_j)
        force_mat = k_repulse * overlap_safe[:, :, np.newaxis] * unit_diffs
        
        # Sum forces for each circle (sum over j)
        pair_forces = np.sum(force_mat, axis=1)
        forces += pair_forces
        
        # Update centers
        centers += dt * forces
        
        # Hard clamp to keep centers within valid bounds [r, 1-r]
        # This prevents numerical drift out of bounds
        lower = radii
        upper = 1.0 - radii
        centers[:, 0] = np.clip(centers[:, 0], lower, upper)
        centers[:, 1] = np.clip(centers[:, 1], lower, upper)
        
        # Decay parameters to settle into local optimum
        if step % 500 == 0:
            growth_rate *= 0.9
            dt *= 0.95
            
            if np.isnan(centers).any() or np.isnan(radii).any():
                break

    # 3. Final Safety Adjustments
    # Ensure strict validity by reducing radii if any violations remain
    
    # Iterative reduction
    for _ in range(100):
        needs_fix = False
        
        # Check boundary constraints
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            min_dist_bound = min(x, 1.0 - x, y, 1.0 - y)
            if r > min_dist_bound + 1e-9:
                radii[i] = max(0.0, min_dist_bound)
                needs_fix = True
        
        # Check pairwise constraints
        # Recompute distances
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Find max violation
        max_viol = -1.0
        viol_pair = (-1, -1)
        
        # Check upper triangle
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                d = dists[i, j]
                sum_r = radii[i] + radii[j]
                if sum_r > d + 1e-9:
                    viol = sum_r - d
                    if viol > max_viol:
                        max_viol = viol
                        viol_pair = (i, j)
        
        if max_viol > 1e-9:
            i, j = viol_pair
            # Reduce radii to satisfy constraint
            # Distribute reduction equally
            reduction = max_viol / 2.0
            radii[i] -= reduction
            radii[j] -= reduction
            # Clamp to non-negative
            if radii[i] < 0: radii[i] = 0.0
            if radii[j] < 0: radii[j] = 0.0
            needs_fix = True
        else:
            break
            
        if not needs_fix:
            break

    # Ensure non-negative radii
    radii = np.maximum(radii, 0.0)
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii