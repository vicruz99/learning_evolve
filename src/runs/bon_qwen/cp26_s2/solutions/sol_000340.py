# sol_000340 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d755ba05) state=18724d57 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a physics-based repulsion simulation with iterative radius expansion.
    """
    n = 26
    
    # 1. Initialize centers in a dense hexagonal-like grid
    centers = np.zeros((n, 2))
    r_init = 0.08
    
    # Generate a hexagonal grid
    # Row spacing: sqrt(3)/2 * diameter = sqrt(3) * r
    # Col spacing: diameter = 2 * r
    idx = 0
    row = 0
    while idx < n:
        # Determine number of circles in this row
        # Hexagonal packing alternates between 6 and 5 to fill space densely
        count_in_row = 6 if row % 2 == 0 else 5
        
        # If we only need a few more circles to reach 26
        remaining = n - idx
        if count_in_row > remaining:
            count_in_row = remaining
        
        # Shift for odd rows
        x_offset = r_init if row % 2 == 1 else 0
        
        for col in range(count_in_row):
            centers[idx, 0] = x_offset + col * (2 * r_init)
            centers[idx, 1] = row * (np.sqrt(3) * r_init)
            idx += 1
        row += 1

    # Center the grid in the unit square initially
    cx, cy = centers.mean(axis=0)
    centers -= np.array([cx - 0.5, cy - 0.5])

    # 2. Optimization Loop
    # We will expand a 'target radius' and adjust centers to accommodate it
    current_r = 0.05
    max_r = 0.12 # Upper bound for equal radii (approx 2.636 / 26)
    
    # Simulation parameters
    dt = 0.1 # Time step
    repulsion_k = 10.0 # Repulsion strength
    damping = 0.8 # Damping factor to prevent oscillation
    
    # Iterate to grow circles
    for step in range(1000):
        # Gradually increase target radius
        target_r = current_r + (max_r - current_r) * (step / 1000.0)
        
        forces = np.zeros_like(centers)
        valid = True
        
        # Calculate forces based on target_r
        for i in range(n):
            # Boundary forces
            for wall_dist, direction in [(centers[i, 0], -1), (1 - centers[i, 0], 1),
                                         (centers[i, 1], -1), (1 - centers[i, 1], 1)]:
                if wall_dist < target_r:
                    overlap = target_r - wall_dist
                    forces[i, 0 if 'x' in str(direction) else 1] += repulsion_k * overlap * direction

            # Inter-circle forces
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                min_dist = 2 * target_r
                
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    force_vec = (diff / dist) * repulsion_k * overlap
                    forces[i] += force_vec
                    forces[j] -= force_vec
        
        # Update positions
        centers += dt * forces
        centers = np.clip(centers, target_r, 1 - target_r)
        
        # Check if system is stable enough to increase radius
        if np.linalg.norm(forces) < 0.1:
            current_r += 0.0005
            if current_r > max_r:
                current_r = max_r

    # 3. Final Polish: Calculate actual radii based on final positions
    # We use the minimum distance to any neighbor or boundary as the radius
    radii = np.ones(n) * current_r
    
    # Adjust radii to ensure strict non-overlap (safe scaling)
    # Find minimum distance between any pair or boundary
    min_dist_pair = np.inf
    min_dist_wall = np.inf
    
    for i in range(n):
        # Wall distances
        d_wall = np.min([centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1]])
        min_dist_wall = min(min_dist_wall, d_wall)
        
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            min_dist_pair = min(min_dist_pair, dist)
            
    # The maximum possible equal radius is half the minimum pairwise distance,
    # constrained by wall distance.
    r_final = min(min_dist_wall, min_dist_pair / 2.0)
    
    # Apply a tiny safety margin for numerical precision
    radii[:] = r_final * 0.9999
    
    # Recalculate sum of radii
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
