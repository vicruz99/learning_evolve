# sol_000048 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b794a107) state=4fa827a5 sum of radii=2.050164 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a force-based repulsion method starting from a hexagonal arrangement.
    """
    np.random.seed(42)
    n = 26
    
    # Initialize positions in a staggered grid (hexagonal-like)
    # We try to fit rows with 5 and 6 circles
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Rough estimation of radius to fit 26 circles
    # Hexagonal packing density suggests r ~ 0.09 - 0.10
    # Let's start with a slightly smaller radius to allow movement
    r_init = 0.08
    
    idx = 0
    row_height = r_init * np.sqrt(3)
    
    # Construct rows: 6, 5, 6, 5, 4 (Total 26)
    # Or 5, 6, 5, 6, 4 (Total 26)
    # Let's try to distribute them to fit width 1 and height 1
    # Row widths: 
    # 6 circles: width approx 6 * 2r = 12r. If r=0.08, width=0.96. Fits.
    # 5 circles: width approx 10r = 0.8. Fits.
    
    rows_config = [6, 5, 6, 5, 4]
    
    current_y = r_init
    for r_count in rows_config:
        # Shift odd rows (index 1, 3) by r_init
        # Determine row index in the list to decide shift
        row_idx = rows_config.index(r_count) # This is not robust if duplicates
        # Let's just keep a counter
        pass

    # Better initialization logic
    centers = np.zeros((n, 2))
    radii[:] = r_init
    
    idx = 0
    y_pos = r_init
    shift = 0.0
    
    # Pattern: 6, 5, 6, 5, 4
    counts = [6, 5, 6, 5, 4]
    
    for i, count in enumerate(counts):
        x_start = 0.5 - (count * 2 * r_init) / 2 + shift
        for j in range(count):
            x = x_start + j * 2 * r_init
            y = y_pos
            centers[idx] = [x, y]
            idx += 1
        
        y_pos += r_init * np.sqrt(3)
        shift = r_init if i % 2 == 0 else 0.0 # Alternate shift

    # Optimization parameters
    iterations = 2000
    step_size = 0.01
    temperature = 0.1
    
    best_sum = 0.0
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    for it in range(iterations):
        # Decay step size and temperature
        current_step = step_size * (1 - it / iterations)
        current_temp = temperature * (1 - it / iterations)
        
        # Calculate forces and potential radius increase
        # We want to expand radii, so we push circles apart
        
        # 1. Calculate max possible radius for each circle based on neighbors and bounds
        # This is a simplified relaxation step
        
        # Check boundaries
        for k in range(n):
            # Distance to boundaries
            dist_x = min(centers[k, 0], 1 - centers[k, 0])
            dist_y = min(centers[k, 1], 1 - centers[k, 1])
            dist_boundary = min(dist_x, dist_y)
            
            # Distance to other circles
            dist_others = np.inf
            for m in range(n):
                if k == m: continue
                d = np.sqrt(np.sum((centers[k] - centers[m])**2))
                if d < dist_others:
                    dist_others = d
            
            # The radius is constrained by (d_ij - r_j) / 2? 
            # Or simply r_k <= dist_others - r_m
            # For optimization, let's just move centers to increase clearance
            
            # Repulsive forces from other circles
            force = np.zeros(2)
            for m in range(n):
                if k == m: continue
                diff = centers[k] - centers[m]
                dist = np.sqrt(np.sum(diff**2))
                if dist < 1e-9: dist = 1e-9
                required_dist = radii[k] + radii[m]
                overlap = required_dist - dist
                if overlap > 0:
                    # Push apart
                    force += (diff / dist) * overlap
            
            # Boundary forces
            if centers[k, 0] - radii[k] < 0:
                force[0] += (0 - (centers[k, 0] - radii[k])) * 10
            if centers[k, 0] + radii[k] > 1:
                force[0] -= (1 - (centers[k, 0] + radii[k])) * 10
            if centers[k, 1] - radii[k] < 0:
                force[1] += (0 - (centers[k, 1] - radii[k])) * 10
            if centers[k, 1] + radii[k] > 1:
                force[1] -= (1 - (centers[k, 1] + radii[k])) * 10
                
            centers[k] += force * current_step
            
            # Clamp to box
            centers[k, 0] = np.clip(centers[k, 0], radii[k], 1 - radii[k])
            centers[k, 1] = np.clip(centers[k, 1], radii[k], 1 - radii[k])

        # Try to increase radii
        # Find the bottleneck for each circle
        # A circle can expand until it hits a boundary or another circle
        # r_k <= x_k, r_k <= 1-x_k, r_k <= y_k, r_k <= 1-y_k
        # r_k + r_m <= dist(k, m) => r_k <= dist(k, m) - r_m
        
        new_radii = radii.copy()
        for k in range(n):
            # Boundary limits
            lim_x = min(centers[k, 0], 1 - centers[k, 0])
            lim_y = min(centers[k, 1], 1 - centers[k, 1])
            limit = min(lim_x, lim_y)
            
            # Neighbor limits
            for m in range(n):
                if k == m: continue
                dist = np.sqrt(np.sum((centers[k] - centers[m])**2))
                limit = min(limit, dist - radii[m])
            
            # We can increase radius if there is slack
            if limit > radii[k]:
                # Increase by a fraction of the available space
                new_radii[k] = radii[k] + 0.1 * (limit - radii[k])
        
        radii = new_radii
        
        # Calculate current sum
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Final validation and cleanup
    # Ensure strict non-overlap by slightly reducing radii if needed
    # But the logic above should maintain it. 
    # Let's do a final check and minor adjustment
    
    for k in range(n):
        # Enforce boundaries strictly
        radii[k] = min(radii[k], centers[k, 0], 1 - centers[k, 0], centers[k, 1], 1 - centers[k, 1])
        
        for m in range(k+1, n):
            dist = np.sqrt(np.sum((centers[k] - centers[m])**2))
            if dist < radii[k] + radii[m]:
                # Reduce larger radius to avoid overlap
                if radii[k] > radii[m]:
                    radii[k] = dist - radii[m]
                else:
                    radii[m] = dist - radii[k]
                # Re-enforce boundaries for m
                radii[m] = min(radii[m], centers[m, 0], 1 - centers[m, 0], centers[m, 1], 1 - centers[m, 1])

    final_sum = np.sum(radii)
    
    return centers, radii, final_sum
