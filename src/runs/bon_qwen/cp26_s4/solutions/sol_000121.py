# sol_000121 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 40ff4175) state=4829ff4d sum of radii=2.013383 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed layout algorithm inspired by hexagonal packing.
    """
    np.random.seed(42)
    n = 26
    best_centers = np.zeros((n, 2))
    best_sum = 0.0
    best_radii = np.zeros(n)

    # Try multiple random shifts of the initial hexagonal lattice to find a good starting point
    for _ in range(10):
        centers = np.zeros((n, 2))
        
        # Hexagonal initialization
        # Row height for hexagonal packing is sqrt(3)/2 * diameter
        # We want to fit 26 circles. 
        # A 5x6 grid is 30 circles, a 5x5 is 25. 
        # We will use a hexagonal arrangement of 26 points.
        
        row_width = 2 * 0.05 # initial guess for radius 0.05, spacing 0.1
        col_height = np.sqrt(3) * 0.05
        count = 0
        r_init = 0.1
        
        # Try to pack in a grid, then scale to fit unit square
        x, y = 0.0, 0.0
        row = 0
        while count < n:
            for i in range(n):
                if count >= n:
                    break
                # Hexagonal offset
                offset = (row % 2) * r_init
                centers[count, 0] = x + i * 2 * r_init + offset
                centers[count, 1] = y + row * col_height
                count += 1
            row += 1
            
        # Normalize to fit in unit square with some margin
        min_c = np.min(centers, axis=0)
        max_c = np.max(centers, axis=0)
        span = max_c - min_c
        scale = 0.9 / np.max(span)
        centers = (centers - min_c) * scale + 0.05
        
        # Add random noise to break symmetry
        centers += np.random.uniform(-0.02, 0.02, size=(n, 2))
        centers = np.clip(centers, 0.02, 0.98)
        
        # Force-directed optimization
        step_size = 0.05
        decay = 0.995
        
        for _ in range(300):
            forces = np.zeros_like(centers)
            radii = np.zeros(n)
            
            # Calculate current radii based on nearest neighbors and boundaries
            for i in range(n):
                min_dist = np.min([
                    centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1]
                ])
                
                for j in range(n):
                    if i != j:
                        d = np.linalg.norm(centers[i] - centers[j])
                        min_dist = min(min_dist, d)
                
                radii[i] = min_dist / 2.0
            
            # Calculate forces (gradient of sum of radii)
            for i in range(n):
                # Force from boundaries (push away from walls)
                # If x is close to 0, force is positive
                if radii[i] == centers[i, 0] / 2.0:
                    forces[i, 0] += step_size
                elif radii[i] == (1.0 - centers[i, 0]) / 2.0:
                    forces[i, 0] -= step_size
                
                if radii[i] == centers[i, 1] / 2.0:
                    forces[i, 1] += step_size
                elif radii[i] == (1.0 - centers[i, 1]) / 2.0:
                    forces[i, 1] -= step_size
                
                # Force from neighbors
                for j in range(n):
                    if i != j:
                        diff = centers[i] - centers[j]
                        dist = np.linalg.norm(diff)
                        if dist < 1e-9: continue
                        if radii[i] == dist / 2.0:
                            # Push i away from j
                            forces[i] += step_size * (diff / dist)
                            
            # Apply forces
            centers += forces
            centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
            step_size *= decay

        # Final calculation of radii
        final_radii = np.zeros(n)
        valid = True
        for i in range(n):
            r = np.min([
                centers[i, 0], 1.0 - centers[i, 0],
                centers[i, 1], 1.0 - centers[i, 1]
            ])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    r = min(r, d)
            final_radii[i] = r / 2.0
            
        current_sum = np.sum(final_radii)
        
        # Check for overlaps explicitly
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < final_radii[i] + final_radii[j] - 1e-9:
                    valid = False
                    break
            if not valid: break
            
        # Check boundaries
        for i in range(n):
            if (centers[i, 0] < final_radii[i] or centers[i, 0] > 1.0 - final_radii[i] or
                centers[i, 1] < final_radii[i] or centers[i, 1] > 1.0 - final_radii[i]):
                valid = False
                break

        if valid and current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = final_radii.copy()

    # Final adjustment to ensure strict non-overlap and boundary conditions
    # Sometimes the optimizer pushes them slightly too close due to numerical issues
    # We can shrink radii slightly if needed, but the logic above should be robust.
    # Just to be safe, we re-calculate radii one last time from centers.
    final_final_radii = np.zeros(n)
    for i in range(n):
        r = np.min([
            best_centers[i, 0], 1.0 - best_centers[i, 0],
            best_centers[i, 1], 1.0 - best_centers[i, 1]
        ])
        for j in range(n):
            if i != j:
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                r = min(r, d)
        final_final_radii[i] = r / 2.0
    
    final_sum = np.sum(final_final_radii)
    
    return best_centers, final_final_radii, final_sum
