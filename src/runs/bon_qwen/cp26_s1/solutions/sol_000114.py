# sol_000114 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4a327247) state=bad4f62c sum of radii=2.499812 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def get_distance_matrix(centers):
    """Compute pairwise Euclidean distance matrix."""
    # centers shape (n, 2)
    # diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    return dists

def calculate_overlap_penalty(centers, radii):
    """Calculate penalty for overlaps between circles."""
    n = len(radii)
    dists = get_distance_matrix(centers)
    # radii sum matrix
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Violation is positive if radii sum > distance
    violation = rad_sum - dists
    # Diagonal is distance 0, radii sum 2r, so positive. We ignore self-interaction.
    np.fill_diagonal(violation, 0)
    
    # Only positive violations penalize
    violation = np.maximum(0, violation)
    
    # Sum of squared violations
    return np.sum(violation**2)

def calculate_boundary_penalty(centers, radii):
    """Calculate penalty for circles going outside [0,1]x[0,1]."""
    penalty = 0.0
    # x < 0 check: center - r >= 0 => r <= center. Violation r - center.
    penalty += np.sum(np.maximum(0, radii - centers[:, 0])**2)
    # x > 1 check: center + r <= 1 => r <= 1 - center. Violation r - (1 - center).
    penalty += np.sum(np.maximum(0, radii - (1.0 - centers[:, 0]))**2)
    # y < 0 check
    penalty += np.sum(np.maximum(0, radii - centers[:, 1])**2)
    # y > 1 check
    penalty += np.sum(np.maximum(0, radii - (1.0 - centers[:, 1]))**2)
    return penalty

def objective_function(variables, n, weight):
    """
    Objective function to minimize: -sum(radii) + weight * penalties
    variables layout: [x0, y0, r0, x1, y1, r1, ...]
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = variables[3 * i]
        centers[i, 1] = variables[3 * i + 1]
        radii[i] = variables[3 * i + 2]
    
    # We want to maximize sum of radii, so minimize negative sum
    obj = -np.sum(radii)
    
    # Add penalties for constraints
    obj += weight * calculate_overlap_penalty(centers, radii)
    obj += weight * calculate_boundary_penalty(centers, radii)
    
    return obj

def get_initial_positions(n):
    """
    Generate initial positions on a hexagonal grid pattern.
    """
    # Estimate radius for a dense packing. 
    # For 26 circles, r approx 0.1.
    r_est = 0.09
    h = r_est * math.sqrt(3)
    
    points = []
    y = r_est
    row_idx = 0
    while y < 1.0:
        row_points = []
        # Hexagonal packing: rows alternate x-offset
        if row_idx % 2 == 0:
            x_start = r_est
        else:
            x_start = 2 * r_est
            
        x = x_start
        while x < 1.0 - r_est + 1e-5:
            # Check if circle fits in x bounds
            if x + r_est <= 1.0 + 1e-9:
                row_points.append([x, y])
            x += 2 * r_est
        
        for p in row_points:
            points.append(p)
            if len(points) >= n:
                break
        if len(points) >= n:
            break
            
        y += h
        row_idx += 1
    
    # Fallback if not enough points generated (unlikely with r=0.09)
    if len(points) < n:
        points = []
        while len(points) < n:
            points.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
            
    return np.array(points[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Main function to pack 26 circles in a unit square.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    best_sum_r = -1.0
    best_centers = None
    best_radii = None
    
    # Prepare a list of initial configurations to try
    configs = []
    
    # 1. Hexagonal grid initialization
    centers_hex = get_initial_positions(n)
    configs.append(centers_hex)
    
    # 2. Random initialization
    centers_rand = np.random.rand(n, 2) * 0.6 + 0.2
    configs.append(centers_rand)
    
    # 3. Square grid initialization (5x5 + 1)
    grid_x = [0.1, 0.3, 0.5, 0.7, 0.9]
    grid_pts = []
    for x in grid_x:
        for y in grid_x:
            grid_pts.append([x, y])
    # Add one extra point
    grid_pts.append([0.5, 0.2]) 
    configs.append(np.array(grid_pts[:n]))
    
    # 4, 5, 6. Additional random restarts
    for _ in range(3):
        configs.append(np.random.rand(n, 2) * 0.6 + 0.2)

    # Penalty weight
    # High weight ensures constraints are respected
    weight = 5000.0
    
    for cfg_centers in configs:
        # Estimate initial radii based on distances
        dists = get_distance_matrix(cfg_centers)
        min_dists = np.min(dists, axis=1)
        
        x = cfg_centers[:, 0]
        y = cfg_centers[:, 1]
        # Distance to nearest wall
        wall_dists = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
        
        # Initialize radii as fraction of available space
        r_init = np.minimum(min_dists, wall_dists) * 0.3
        r_init = np.maximum(r_init, 0.01)
        
        # Flatten to optimization variables [x0, y0, r0, x1, y1, r1, ...]
        variables = np.zeros(3 * n)
        for i in range(n):
            variables[3*i] = cfg_centers[i, 0]
            variables[3*i+1] = cfg_centers[i, 1]
            variables[3*i+2] = r_init[i]
        
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for _ in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
            
        try:
            # Optimize
            res = opt.minimize(
                objective_function, 
                variables, 
                method='L-BFGS-B', 
                bounds=bounds, 
                args=(n, weight),
                options={'maxiter': 3000, 'ftol': 1e-12}
            )
            
            # Extract solution
            c = res.x[0::3]
            cy = res.x[1::3]
            r = res.x[2::3]
            
            res_centers = np.column_stack((c, cy))
            res_radii = r
            
            # Post-processing to strictly satisfy constraints
            
            # 1. Clamp radii to satisfy boundary constraints
            for i in range(n):
                rx, ry = res_centers[i]
                r_val = res_radii[i]
                # Max radius allowed by walls
                max_r_wall = min(rx, 1-rx, ry, 1-ry)
                res_radii[i] = min(r_val, max_r_wall)
                res_radii[i] = max(0.0, res_radii[i])
            
            # 2. Iteratively shrink radii to resolve overlaps
            for _ in range(30):
                dists_mat = get_distance_matrix(res_centers)
                rad_sum = res_radii[:, np.newaxis] + res_radii[np.newaxis, :]
                violation = rad_sum - dists_mat
                np.fill_diagonal(violation, 0)
                max_v = np.max(violation)
                
                if max_v < 1e-9:
                    break
                
                # Find the pair with maximum overlap
                idx = np.unravel_index(np.argmax(violation), violation.shape)
                i, j = idx
                
                # Shrink both radii to resolve overlap
                # Shrink amount is half the overlap plus a small margin
                shrink = max_v / 2.0 + 1e-7
                res_radii[i] = max(0.0, res_radii[i] - shrink)
                res_radii[j] = max(0.0, res_radii[j] - shrink)
                
            # Calculate final sum
            current_sum = np.sum(res_radii)
            
            if current_sum > best_sum_r:
                best_sum_r = current_sum
                best_centers = res_centers.copy()
                best_radii = res_radii.copy()
                
        except Exception:
            continue
            
    # Fallback if no solution found (should not happen)
    if best_centers is None:
        best_centers = get_initial_positions(n)
        best_radii = np.ones(n) * 0.01
        best_sum_r = np.sum(best_radii)

    return best_centers, best_radii, best_sum_r
