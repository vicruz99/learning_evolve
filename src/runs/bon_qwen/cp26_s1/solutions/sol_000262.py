# sol_000262 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fd8f28d8) state=e736cb75 sum of radii=2.423699 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def generate_hexagonal_initialization(n_circles, seed=None):
    """
    Generates an initial placement of n_circles in a unit square 
    using a hexagonal lattice pattern.
    """
    if seed is not None:
        np.random.seed(seed)

    # Estimate spacing. For 26 circles, approx 5x5 grid density.
    # Hexagonal packing density is higher. 
    # We create a grid denser than needed and pick points, or calculate exact fit.
    # Let's try to fit them in rows.
    
    # Heuristic number of rows/cols based on area
    # Area per circle approx 1/26. r approx 0.1. Diameter 0.2.
    # Spacing approx 0.2.
    
    spacing = 0.2
    points = []
    
    # Try to fill rows
    y = 0
    row_idx = 0
    while len(points) < n_circles:
        x_offset = (spacing / 2) if (row_idx % 2 == 1) else 0
        x = x_offset
        
        while x < 1 + spacing/2 and len(points) < n_circles:
            if x <= 1:
                points.append([x, y])
            x += spacing
        y += spacing * np.sqrt(3) / 2
        row_idx += 1
        
    # If we generated fewer than needed (unlikely with loop condition, but safe), 
    # or if we need to trim, handle it. 
    # The loop condition ensures we stop exactly when filled or grid ends.
    # However, if grid runs out, we might have fewer.
    # Let's ensure we have exactly n_circles.
    
    # If we ran out of space in the grid generation (unlikely for 26), 
    # fallback to random.
    if len(points) < n_circles:
        points = np.random.uniform(0.1, 0.9, size=(n_circles, 2)).tolist()
        
    # If we generated more, take first n
    points = points[:n_circles]
    
    # Normalize to fit strictly inside [0,1] with some padding for radii
    # Center the grid
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    # Shift to center if needed, or just ensure bounds
    # For simplicity, just return coordinates, optimizer will adjust.
    # But let's ensure they are in [0,1].
    coords = np.array(points)
    
    # Add small random noise to break symmetry
    coords += np.random.uniform(-0.01, 0.01, size=coords.shape)
    
    # Clip to valid range for centers [0.05, 0.95] roughly
    coords = np.clip(coords, 0.05, 0.95)
    
    return coords

def penalty_function(params, n_circles):
    """
    Calculates the objective value: negative sum of radii + penalty for violations.
    """
    # Reshape params: 78 -> 26 x 3 (x, y, r)
    data = params.reshape((n_circles, 3))
    centers = data[:, :2]
    radii = data[:, 2]
    
    # Objective: -sum(radii)
    obj = -np.sum(radii)
    
    # Penalty weight
    mu = 1000.0
    
    # 1. Pairwise overlap penalties
    # Compute distance matrix
    # dist_matrix[i, j] = distance between i and j
    # Using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    # Triangular mask to avoid double counting and self
    # We only care about i < j
    # Overlap condition: dist < r_i + r_j => violation = (r_i + r_j) - dist
    # We want max(0, violation)
    
    r_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Strictly lower triangle indices
    rows, cols = np.tril_indices(n_circles, k=-1)
    
    violations_dist = r_sum_matrix[rows, cols] - dist_matrix[rows, cols]
    violations_dist = np.maximum(0, violations_dist)
    
    overlap_penalty = mu * np.sum(violations_dist**2)
    
    # 2. Boundary penalties
    # x >= r  => r - x <= 0
    # x <= 1-r => x + r - 1 <= 0
    # Same for y
    
    # Left/Right walls
    v_left = radii - centers[:, 0]
    v_right = radii + centers[:, 0] - 1.0
    
    # Bottom/Top walls
    v_bottom = radii - centers[:, 1]
    v_top = radii + centers[:, 1] - 1.0
    
    wall_violations = np.concatenate([v_left, v_right, v_bottom, v_top])
    wall_violations = np.maximum(0, wall_violations)
    
    wall_penalty = mu * np.sum(wall_violations**2)
    
    return obj + overlap_penalty + wall_penalty

def get_valid_radii(centers, radii):
    """
    Adjusts radii to ensure strict validity based on fixed centers.
    Returns valid radii array.
    """
    n = len(radii)
    # Current radii might be invalid. We need to reduce them.
    # A simple approach: compute the max possible radius for each circle 
    # given the others, but that's iterative.
    # A safe approach: scale all radii down uniformly until valid.
    
    # Check max violation
    max_violation = 0.0
    
    # Check walls
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Distance to walls
        d_wall = min(x, 1-x, y, 1-y)
        if r > d_wall:
            # Need to reduce r to d_wall
            # But we scale globally?
            # Let's just clamp individually for a quick fix if needed, 
            # but uniform scaling is safer for sum.
            pass # Handled by scaling factor below
        
    # Compute required scaling factor
    # Scale factor s such that s*r_i + s*r_j <= dist_ij
    # s <= dist_ij / (r_i + r_j)
    # And s*r_i <= dist_to_wall
    
    s = 1.0
    
    # Wall constraints
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r > 0:
            d_wall = min(x, 1-x, y, 1-y)
            req_s = d_wall / r
            if req_s < s:
                s = req_s
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i+1, n):
            r_sum = radii[i] + radii[j]
            if r_sum > 0:
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                req_s = dist / r_sum
                if req_s < s:
                    s = req_s
    
    # Apply scaling
    # Add small epsilon to be strictly inside
    s = max(0.0, s - 1e-6)
    
    valid_radii = radii * s
    return valid_radii

def run_packing():
    n_circles = 26
    best_sum_radii = -1.0
    best_result = None
    
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    # Actually r can be up to 0.5, but practically smaller.
    bounds = []
    for _ in range(n_circles):
        bounds.extend([
            (0.0, 1.0), # x
            (0.0, 1.0), # y
            (0.0, 0.5)  # r
        ])
        
    # Run multiple times with different seeds
    for seed in range(10):
        # Initialization
        # Use hexagonal grid perturbed by seed
        # Generate a base grid
        # Since we can't use lambda, we inline or use helper
        # Helper generate_hexagonal_initialization is defined above
        
        # We need to pass seed to generator if it uses random
        # But generate_hexagonal_initialization uses np.random.seed
        
        # Create initial positions
        centers_init = generate_hexagonal_initialization(n_circles, seed=seed)
        # Initial radii: small valid radius
        # Estimate max possible radius based on center density? 
        # Just pick 0.05
        radii_init = np.full(n_circles, 0.05)
        
        # Combine into params array
        x0 = np.empty(n_circles * 3)
        for i in range(n_circles):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = radii_init[i]
            
        # Optimization
        # We minimize the penalty function
        res = minimize(
            penalty_function, 
            x0, 
            args=(n_circles,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        
        # Extract results
        params_opt = res.x
        centers_opt = params_opt.reshape((n_circles, 3))[:, :2]
        radii_opt = params_opt.reshape((n_circles, 3))[:, 2]
        
        # Post-process to ensure validity
        radii_valid = get_valid_radii(centers_opt, radii_opt)
        current_sum = np.sum(radii_valid)
        
        # Check validity explicitly
        # (Simplified check, get_valid_radii should ensure it)
        # But let's be safe.
        
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_result = (centers_opt, radii_valid)
            
    centers, radii = best_result
    return centers, radii, np.sum(radii)

# Note: The problem requires the function run_packing to be defined.
# The helper functions are defined at top level.
