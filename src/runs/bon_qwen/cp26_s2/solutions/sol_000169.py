# sol_000169 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d2f1ce33) state=b0e88caf sum of radii=1.337401 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Number of circles
    n = 26
    
    # 1. Initialize Centers and Radii
    # We place 25 circles in a 5x5 grid and 1 circle in the center gap.
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.08) # Start with small radii to allow movement
    
    # 5x5 grid initialization
    grid_idx = 0
    for i in range(5):
        for j in range(5):
            x = 0.1 + j * 0.2
            y = 0.1 + i * 0.2
            centers[grid_idx] = [x, y]
            grid_idx += 1
            
    # 26th circle in the center
    centers[25] = [0.5, 0.5]

    # 2. Force-Directed Optimization
    # We treat this as a physics simulation where circles repel each other.
    # We use a high number of iterations to allow the system to settle.
    
    lr = 0.005  # Learning rate / step size
    repulsion_strength = 1.0
    
    for step in range(5000):
        forces = np.zeros_like(centers)
        valid = True
        
        for i in range(n):
            for j in range(i + 1, n):
                # Vector from i to j
                vec = centers[j] - centers[i]
                dist = np.linalg.norm(vec)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Overlap detected: repel
                    # Force proportional to overlap amount
                    overlap = min_dist - dist
                    if dist > 1e-9:
                        dir_vec = vec / dist
                        forces[i] -= dir_vec * overlap * repulsion_strength
                        forces[j] += dir_vec * overlap * repulsion_strength
                    else:
                        # If centers coincide, push apart randomly slightly
                        forces[i] -= (np.random.rand(2) - 0.5)
                        forces[j] += (np.random.rand(2) - 0.5)
        
        # Boundary constraints (push inward if outside)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (0 - (x - r))
            # Right wall
            if x + r > 1:
                forces[i, 0] -= ((x + r) - 1)
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (0 - (y - r))
            # Top wall
            if y + r > 1:
                forces[i, 1] -= ((y + r) - 1)
                
        # Update positions
        centers += forces * lr
        
        # Clip positions to stay within [0,1] strictly to prevent divergence
        centers = np.clip(centers, 1e-6, 1 - 1e-6)
        
        # Gradually increase radii to force optimization
        # We increase radii slowly so the repulsion forces have time to resolve
        current_r_increase = 0.00002
        radii += current_r_increase

    # 3. Local Optimization to find exact max radii
    # Once positions are roughly optimal, we solve for the exact max radii.
    # We can do this by solving a series of linear constraints or just 
    # iterating the radii expansion tightly.
    
    # Tighten radii based on current positions
    # Max radius for circle i is min(dist(i, j) - r_j) for all j, and boundary distances
    # Since we want to maximize sum, we can use a relaxation method.
    
    # Simple iterative solver for radii given fixed centers
    for _ in range(1000):
        # Calculate max possible radius for each circle based on current others
        new_radii = np.zeros(n)
        for i in range(n):
            # Start with boundary limits
            max_r = min(centers[i, 0], 1 - centers[i, 0], 
                        centers[i, 1], 1 - centers[i, 1])
            
            for j in range(n):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                # r_i + r_j <= dist  => r_i <= dist - r_j
                # We use current r_j to estimate constraint
                limit = dist - radii[j]
                if limit < max_r:
                    max_r = limit
            
            # Smooth update to avoid oscillation, but push towards limit
            # Using a weighted average helps stability
            new_radii[i] = max_r
        
        # Update radii, ensuring they don't shrink drastically if valid
        # We want to maximize, so we take the calculated max_r
        radii = new_radii

    # Ensure radii are non-negative
    radii = np.maximum(radii, 0)

    # 4. Final Validation and Adjustment
    # Re-run the physics simulation briefly with fixed radii to resolve any final overlaps
    # caused by the discrete update of radii.
    lr_final = 0.001
    for _ in range(1000):
        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                vec = centers[j] - centers[i]
                dist = np.linalg.norm(vec)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if dist > 1e-9:
                        dir_vec = vec / dist
                        forces[i] -= dir_vec * overlap
                        forces[j] += dir_vec * overlap
            # Boundary forces
            x, y = centers[i]
            r = radii[i]
            if x - r < 0: forces[i, 0] += (r - x)
            if x + r > 1: forces[i, 0] -= (x + r - 1)
            if y - r < 0: forces[i, 1] += (r - y)
            if y + r > 1: forces[i, 1] -= (y + r - 1)
        centers += forces * lr_final
        centers = np.clip(centers, 1e-9, 1 - 1e-9)
        
        # Recalculate radii based on new tight positions
        for i in range(n):
            max_r = min(centers[i, 0], 1 - centers[i, 0], 
                        centers[i, 1], 1 - centers[i, 1])
            for j in range(n):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                max_r = min(max_r, dist - radii[j])
            radii[i] = max_r
            
    # Final clip to ensure no negative radii
    radii = np.maximum(radii, 0.0)

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
