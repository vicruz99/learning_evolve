# sol_000014 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 60d0e48a) state=7d44c9de sum of radii=2.615271 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def run_packing():
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # We will optimize centers and radii directly using SLSQP
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Total 3 * n variables.
    
    # Helper to create a good initial guess
    def get_initial_guess(rotation_angle=0.0):
        # Generate a hexagonal lattice subset
        # We want a compact cluster of 26 points
        # Let's try a roughly 5x5 hexagonal shape
        # Rows with 5, 5, 5, 5, 6 points? Or similar.
        # Let's generate a larger grid and pick closest to center
        
        points = []
        # Lattice vectors for hex packing with spacing 1 (will scale later)
        # u = (1, 0), v = (0.5, sqrt(3)/2)
        # We generate a range of indices
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = i + 0.5 * j
                y = math.sqrt(3)/2 * j
                points.append([x, y])
        
        points = np.array(points)
        
        # Rotate points
        c = math.cos(rotation_angle)
        s = math.sin(rotation_angle)
        rotated_points = np.zeros_like(points)
        rotated_points[:, 0] = c * points[:, 0] - s * points[:, 1]
        rotated_points[:, 1] = s * points[:, 0] + c * points[:, 1]
        
        # Center the points
        center = np.mean(rotated_points, axis=0)
        rotated_points -= center
        
        # We need to select 26 points.
        # Selecting the 26 points closest to the origin gives a compact cluster.
        dists = np.sqrt(rotated_points[:, 0]**2 + rotated_points[:, 1]**2)
        indices = np.argsort(dists)[:n]
        selected_points = rotated_points[indices]
        
        # Scale to fit in unit square roughly
        # Current bounding box
        x_min, x_max = np.min(selected_points[:, 0]), np.max(selected_points[:, 0])
        y_min, y_max = np.min(selected_points[:, 1]), np.max(selected_points[:, 1])
        width = x_max - x_min
        height = y_max - y_min
        max_dim = max(width, height)
        
        # Scale so that it fits in 0.8x0.8 (leaving room for radii)
        scale = 0.7 / max_dim
        selected_points *= scale
        
        # Center in [0,1]
        center = np.mean(selected_points, axis=0)
        selected_points -= center
        selected_points += 0.5
        
        # Initialize radii
        # Start with a reasonable radius, e.g., 0.1
        # But ensure they don't overlap initially.
        # With scale 0.7/max_dim, the inter-point distance is preserved ratio.
        # Original lattice spacing 1.
        # Scaled spacing = scale.
        # Radius should be < scale/2.
        # Let's pick r = scale * 0.2 (very small) to start safe, optimizer will expand.
        # Or estimate based on grid. 5x5 grid fits r=0.1.
        # Our cluster is similar.
        r_init = 0.05 
        
        # Construct variable vector
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = selected_points[i, 0]
            x0[3*i+1] = selected_points[i, 1]
            x0[3*i+2] = r_init
            
        return x0

    # Objective function: Negate sum of radii (since we minimize)
    def objective(vars):
        return -np.sum(vars[2::3])

    # Constraints
    # 1. Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    # 2. Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # 3. r >= 0
    
    def constraint_boundary(vars):
        cons = []
        for i in range(n):
            x = vars[3*i]
            y = vars[3*i+1]
            r = vars[3*i+2]
            
            # x - r >= 0  =>  x - r
            cons.append(x - r)
            # x + r <= 1  =>  1 - (x + r)
            cons.append(1 - (x + r))
            # y - r >= 0
            cons.append(y - r)
            # y + r <= 1
            cons.append(1 - (y + r))
        return np.array(cons)

    def constraint_overlap(vars):
        cons = []
        for i in range(n):
            xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
            for j in range(i + 1, n):
                xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
                
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                sum_r = ri + rj
                
                # dist >= sum_r  => dist^2 >= sum_r^2
                # But squaring can introduce issues if dist < 0? No, dist^2 is positive.
                # However, dist^2 >= sum_r^2 is equivalent to dist >= sum_r since both non-negative.
                # Actually, if sum_r is negative (not possible here), it would be different.
                # ri, rj >= 0.
                cons.append(dist_sq - sum_r**2)
        return np.array(cons)
    
    def constraint_radius_positive(vars):
        return vars[2::3] # r >= 0

    # Define constraints for SLSQP
    # SLSQP expects constraints as dicts or functions returning arrays (>= 0)
    # We will combine them or pass separately.
    # Combining might be cleaner for the function signature, but passing list of dicts is standard.
    
    cons = []
    
    # Boundary constraints
    cons.append({
        'type': 'ineq',
        'fun': constraint_boundary
    })
    
    # Overlap constraints
    cons.append({
        'type': 'ineq',
        'fun': constraint_overlap
    })
    
    # Radius positive (handled by bounds usually, but SLSQP supports bounds)
    # We'll use bounds for x, y, r
    # x, y in [0, 1] (actually tighter with r)
    # r in [0, 0.5]
    
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    # Optimization strategy: Try a few rotations and pick best
    best_result = None
    best_sum = -np.inf
    best_vars = None
    
    # Angles to try
    angles = [0.0, np.pi/6, np.pi/4, np.pi/3, 0.1, 0.2, 0.3]
    
    for angle in angles:
        try:
            x0 = get_initial_guess(angle)
            
            # Use SLSQP
            # maxiter might need to be high
            res = scipy.optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 500, 'ftol': 1e-9}
            )
            
            if res.success or (res.fun < -2.0): # heuristic check
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_vars = res.x.copy()
                    best_result = res
        except Exception as e:
            # print(f"Error with angle {angle}: {e}")
            pass

    if best_vars is None:
        # Fallback: simple grid
        # 5x5 grid is 25. We need 26.
        # Just return a valid packing, maybe smaller radii.
        # This should not happen if optimization works.
        pass

    # Extract centers and radii
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    if best_vars is not None:
        for i in range(n):
            centers[i, 0] = best_vars[3*i]
            centers[i, 1] = best_vars[3*i+1]
            radii[i] = best_vars[3*i+2]
    else:
        # Fallback initialization if optimization failed completely
        # 5x5 grid + 1 small circle?
        # Just return empty or zeros? No, must be valid.
        # Let's create a sparse grid.
        k = 6 # 6x6 = 36 circles, radius 1/12 = 0.0833
        r = 1.0 / (2 * 6)
        idx = 0
        for i in range(6):
            for j in range(6):
                if idx < n:
                    centers[idx, 0] = (2*i + 1) * r
                    centers[idx, 1] = (2*j + 1) * r
                    radii[idx] = r
                    idx += 1

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
