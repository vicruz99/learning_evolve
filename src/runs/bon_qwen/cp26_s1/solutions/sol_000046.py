# sol_000046 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ff99986a) state=aa6a7dfb sum of radii=2.101615 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    Uses a jostling algorithm on a hexagonal lattice initialization.
    """
    n_circles = 26
    centers = np.zeros((n_circles, 2))
    
    # 1. Initialization: Hexagonal Lattice
    # We try to fit 26 circles. A 5x5 grid is 25. 
    # We can use a hexagonal arrangement to pack them tighter.
    # Approximate hexagonal spacing.
    # Let's place them in rows. 
    # Row counts could be 5, 5, 5, 5, 6 (sum 26) or similar.
    # Let's try to place them roughly uniformly first, then the jostling will optimize.
    # A dense grid placement is safer to ensure initial validity.
    
    # Using a grid for initialization is robust. 
    # 6 columns, 5 rows (30 slots), pick first 26? 
    # Or just 6x5 grid and let optimizer shrink? No, we want them large.
    # Let's place them in a 5x5 grid (25) and one in the center?
    # Actually, random perturbation of a grid often works well.
    
    # Let's try a structured hexagonal init.
    # Rows at y = 0.5, 0.5 +/- h, ...
    # Let's just use a dense uniform grid first.
    rows = 5
    cols = 6 # 5*6 = 30, we use 26
    idx = 0
    y_coords = np.linspace(0.5/rows, 1 - 0.5/rows, rows)
    x_coords = np.linspace(0.5/cols, 1 - 0.5/cols, cols)
    
    # Fill centers
    for r in range(rows):
        for c in range(cols):
            if idx < n_circles:
                centers[idx, 0] = x_coords[c]
                centers[idx, 1] = y_coords[r]
                idx += 1
            else:
                break
        if idx >= n_circles:
            break
            
    # Apply a slight random perturbation to break symmetry
    np.random.seed(42)
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    
    # Clip to initial valid range (very small r)
    r = 0.01
    centers = np.clip(centers, r, 1 - r)

    # 2. Jostling Loop
    growth_rate = 1.005
    max_resolutions = 200
    max_growth_steps = 2000
    
    for step in range(max_growth_steps):
        # Try to increase radius
        r_new = r * growth_rate
        
        # Check if this radius is feasible by trying to resolve overlaps
        # We will simulate the resolution
        
        current_centers = centers.copy()
        overlaps = 0
        max_iters = 50 # Max resolution passes per growth step
        
        # Resolve overlaps for r_new
        for _ in range(max_iters):
            collision_found = False
            # Check all pairs
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    dx = current_centers[i, 0] - current_centers[j, 0]
                    dy = current_centers[i, 1] - current_centers[j, 1]
                    dist_sq = dx*dx + dy*dy
                    min_dist = 2 * r_new
                    if dist_sq < min_dist*min_dist and dist_sq > 1e-12:
                        dist = np.sqrt(dist_sq)
                        # Overlap amount
                        overlap = min_dist - dist
                        # Displacement vector (normalize)
                        nx = dx / dist
                        ny = dy / dist
                        
                        # Move apart by half overlap
                        shift_x = nx * overlap * 0.5
                        shift_y = ny * overlap * 0.5
                        
                        current_centers[i, 0] += shift_x
                        current_centers[i, 1] += shift_y
                        current_centers[j, 0] -= shift_x
                        current_centers[j, 1] -= shift_y
                        collision_found = True
            
            if not collision_found:
                break
        
        # Check boundary constraints and fix
        # Also count remaining overlaps
        remaining_overlaps = 0
        for i in range(n_circles):
            # Boundary constraints
            if current_centers[i, 0] < r_new:
                current_centers[i, 0] = r_new
            elif current_centers[i, 0] > 1 - r_new:
                current_centers[i, 0] = 1 - r_new
            
            if current_centers[i, 1] < r_new:
                current_centers[i, 1] = r_new
            elif current_centers[i, 1] > 1 - r_new:
                current_centers[i, 1] = 1 - r_new

        # Re-check overlaps after boundary fix (boundaries might push them into each other)
        # Do a quick pass
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = current_centers[i, 0] - current_centers[j, 0]
                dy = current_centers[i, 1] - current_centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 2 * r_new - 1e-12:
                    remaining_overlaps += 1
        
        if remaining_overlaps == 0:
            # Successfully resolved, accept new radius
            r = r_new
            centers = current_centers
        else:
            # Cannot resolve, reduce radius slightly and stop
            r = r / growth_rate
            break

    # Final validation and adjustment
    # Ensure strict validity
    radii = np.full(n_circles, r)
    
    # Final boundary clamp
    for i in range(n_circles):
        centers[i, 0] = max(r, min(1 - r, centers[i, 0]))
        centers[i, 1] = max(r, min(1 - r, centers[i, 1]))

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
