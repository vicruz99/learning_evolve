# sol_000079 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1daf7277) state=2534a0f1 sum of radii=1.415199 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    # Initialize centers and radii
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    # Initial radii estimation
    r_base = 0.1
    
    # Initialize 5 core circles (larger)
    # 1 center
    centers[0] = [0.5, 0.5]
    radii[0] = 0.25 # Tentative large radius
    
    # 4 corners
    corner_offsets = [
        [0.25, 0.25], [0.75, 0.25], 
        [0.25, 0.75], [0.75, 0.75]
    ]
    for i in range(1, 5):
        centers[i] = corner_offsets[i-1]
        radii[i] = 0.20 # Slightly smaller than center
        
    # Initialize 16 surrounding circles (smaller)
    idx = 5
    # 4 midpoints of sides
    midpoints = [
        [0.5, 0.15], [0.15, 0.5], [0.5, 0.85], [0.85, 0.5]
    ]
    for i, m in enumerate(midpoints):
        centers[idx + i] = m
        radii[idx + i] = 0.08
        
    # 12 additional circles in the gaps
    # Approximate positions based on hexagonal packing logic
    additional_pos = [
        [0.20, 0.20], [0.80, 0.20], [0.20, 0.80], [0.80, 0.80], # Near corners but inside
        [0.35, 0.35], [0.65, 0.35], [0.35, 0.65], [0.65, 0.65], # Near center
        [0.10, 0.35], [0.90, 0.35], [0.10, 0.65], [0.90, 0.65]  # Near edges
    ]
    for i in range(12):
        centers[idx + 4 + i] = additional_pos[i]
        radii[idx + 4 + i] = 0.06

    # Optimization parameters
    iterations = 2000
    lr = 0.01 # Learning rate for positions
    r_lr = 0.001 # Learning rate for radii
    
    # Optimization loop
    for _ in range(iterations):
        # Calculate repulsive forces between circles
        forces = np.zeros_like(centers)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist_vec = centers[i] - centers[j]
                dist = np.linalg.norm(dist_vec)
                min_dist = radii[i] + radii[j]
                
                if dist < 1e-8:
                    dist = 1e-8
                    dist_vec = np.random.rand(2) * 1e-4
                
                if dist < min_dist:
                    # Overlap penalty
                    overlap = min_dist - dist
                    force_mag = overlap / dist
                    forces[i] += force_mag * dist_vec
                    forces[j] -= force_mag * dist_vec

        # Calculate boundary repulsion
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # X boundaries
            if x - r < 0:
                forces[i, 0] += (r - x)
            if x + r > 1:
                forces[i, 0] -= (x + r - 1)
                
            # Y boundaries
            if y - r < 0:
                forces[i, 1] += (r - y)
            if y + r > 1:
                forces[i, 1] -= (y + r - 1)

        # Update positions
        centers += lr * forces
        
        # Clamp positions to [0, 1]
        centers = np.clip(centers, 0, 1)
        
        # Update radii
        # Try to increase radii if no overlap, decrease if overlap
        for i in range(n):
            x, y = centers[i]
            # Max possible radius from boundaries
            max_r_boundary = min(x, 1-x, y, 1-y)
            
            # Check distances to other circles
            max_r_neighbors = np.inf
            for j in range(n):
                if i != j:
                    dist = np.linalg.norm(centers[i] - centers[j])
                    r_j = radii[j]
                    # r_i <= dist - r_j
                    if dist - r_j < max_r_neighbors:
                        max_r_neighbors = dist - r_j
            
            max_r = min(max_r_boundary, max_r_neighbors)
            
            # Gradient ascent for radius
            # If max_r > current_r, we can increase.
            # To be safe and converge, we move r towards max_r
            if max_r > radii[i]:
                radii[i] += r_lr * (max_r - radii[i])
            else:
                radii[i] = max(0, max_r) # If overlap, reduce to max possible

    # Final validation and adjustment to ensure no NaNs or negative radii
    radii = np.clip(radii, 0, None)
    centers = np.clip(centers, 0, 1)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
