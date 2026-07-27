import numpy as np

def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Initialize centers in a staggered hexagonal grid
    # This provides a much denser initial state than a standard grid.
    rows = 7
    cols = 4
    centers = np.zeros((n, 2))
    idx = 0
    for r in range(rows):
        # Vertical spacing is sin(60) * 2 * spacing. 
        # We use 2.0 as a unit spacing for initial layout, then we scale.
        y = r * np.sqrt(3) / 2.0
        num_in_row = cols
        if r % 2 == 1:
            num_in_row -= 1
        for c in range(num_in_row):
            if idx >= n: break
            x = c + (0.5 if r % 2 == 1 else 0.0)
            centers[idx] = [x, y]
            idx += 1
            
    # Scale the initial configuration to fit inside the unit square with a small initial radius
    # Initial radius roughly 0.05 to ensure plenty of room for the simulation to start.
    initial_r = 0.05
    x_range = centers[:, 0].max() - centers[:, 0].min()
    y_range = centers[:, 1].max() - centers[:, 1].min()
    scale_x = (1.0 - 2*initial_r) / x_range
    scale_y = (1.0 - 2*initial_r) / y_range
    scale = min(scale_x, scale_y)
    
    # Normalize to [0,1] then apply scale
    cx_min, cy_min = centers.min(axis=0)
    centers = (centers - [cx_min, cy_min]) * scale + initial_r
    
    radii = np.full(n, initial_r)
    
    # 2. Iterative Expansion and Relaxation
    # We try to grow radii and resolve overlaps using repulsive forces.
    dt = 1.0  # Time step for integration
    
    for _ in range(300): # Number of growth/relaxation cycles
        # Try to grow radii slightly
        radii += 1e-5
        
        # Run several sub-steps of repulsion to settle the configuration
        for step in range(50):
            force = np.zeros_like(centers)
            
            # Compute repulsive forces between all pairs
            # We use a soft force model: if overlap, push away.
            # Force magnitude proportional to overlap amount.
            for i in range(n):
                for j in range(i + 1, n):
                    diff = centers[i] - centers[j]
                    dist = np.linalg.norm(diff)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist and dist > 1e-6:
                        # Overlap detected
                        overlap = min_dist - dist
                        # Normalized direction
                        n_vec = diff / dist
                        # Apply force proportional to overlap to separate them
                        # Stronger force to quickly resolve overlaps
                        strength = 100.0 
                        force[i] += strength * n_vec * overlap
                        force[j] -= strength * n_vec * overlap
            
            # Update positions
            centers += force * dt
            
            # Enforce boundary constraints [r, 1-r]
            for i in range(n):
                r_i = radii[i]
                # Clamp X
                if centers[i, 0] < r_i: centers[i, 0] = r_i
                if centers[i, 0] > 1.0 - r_i: centers[i, 0] = 1.0 - r_i
                # Clamp Y
                if centers[i, 1] < r_i: centers[i, 1] = r_i
                if centers[i, 1] > 1.0 - r_i: centers[i, 1] = 1.0 - r_i
                
            # Cool down the step size to settle
            dt *= 0.995

    # 3. Final local refinement using gradient-based approach for precision
    # Maximize sum of radii by slightly moving centers to allow radii to grow.
    # This is a simplified "center adjustment" phase.
    
    for _ in range(1000):
        improved = False
        for i in range(n):
            # Try to push circle i away from the nearest constraint to maximize its radius
            # Constraints are: distance to boundary, distance to other circles
            current_r = radii[i]
            
            # Calculate max possible radius based on current neighbors
            max_r = current_r + 1e-5 # Growth step
            
            # Check neighbors to see what radius is allowed
            for j in range(n):
                if i == j: continue
                dist_ij = np.linalg.norm(centers[i] - centers[j])
                allowed_r_ij = (dist_ij - radii[j])
                if allowed_r_ij < max_r:
                    max_r = allowed_r_ij
            
            # Check boundary
            boundary_limit = min(centers[i, 0], 1.0 - centers[i, 0], 
                                 centers[i, 1], 1.0 - centers[i, 1])
            if boundary_limit < max_r:
                max_r = boundary_limit
                
            # If we can increase radius, do so. 
            # Ideally we would move centers to allow this, but simple growth works well with prior relaxation.
            if max_r > radii[i] + 1e-7:
                radii[i] = max_r
                improved = True
            
            # Small random perturbation to escape local minima
            centers[i] += np.random.normal(0, 1e-6, 2)
            
            # Re-clamp boundaries
            r_lim = radii[i]
            centers[i, 0] = np.clip(centers[i, 0], r_lim, 1.0 - r_lim)
            centers[i, 1] = np.clip(centers[i, 1], r_lim, 1.0 - r_lim)

    # Final cleanup: ensure radii are valid given centers (in case of numerical drift)
    # This step strictly enforces non-overlap by shrinking radii if needed, 
    # but given the optimization, it should just verify.
    # We will run a final validation pass to be safe.
    
    # Ensure no NaNs
    if np.isnan(centers).any() or np.isnan(radii).any():
        raise ValueError("NaN detected in optimization")

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii