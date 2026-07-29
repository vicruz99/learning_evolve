# sol_000093 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8101c7b4) state=3053598e sum of radii=2.009812 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars, n):
    """
    Objective function to minimize.
    Minimizes -sum(radii) + penalty for constraints.
    
    Args:
        vars: np.array of shape (3*n,) containing [x0, y0, ..., xn, yn, r0, ..., rn]
        n: int, number of circles
        
    Returns:
        float: objective value
    """
    # Extract centers and radii
    # vars layout: first 2*n elements are centers (x, y), last n are radii
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Objective: Maximize sum of radii => Minimize -sum(radii)
    obj = -np.sum(r)
    
    penalty = 0.0
    weight = 2000.0
    
    # Boundary penalties
    # Constraints: r <= x <= 1-r  =>  x - r >= 0  and  1 - x - r >= 0
    # Violations: r - x > 0, x + r - 1 > 0
    # Same for y
    
    x = c[:, 0]
    y = c[:, 1]
    
    # Vectorized boundary penalties
    p_bound = 0.0
    p_bound += np.sum(np.maximum(0, r - x)**2)
    p_bound += np.sum(np.maximum(0, x + r - 1)**2)
    p_bound += np.sum(np.maximum(0, r - y)**2)
    p_bound += np.sum(np.maximum(0, y + r - 1)**2)
    penalty += p_bound
    
    # Overlap penalties
    # Constraints: dist(i, j) >= r_i + r_j
    # Violation: r_i + r_j - dist > 0
    
    # Compute distance matrix efficiently
    # diff shape (n, n, 2)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Ignore diagonal (self-distance) by setting to infinity
    # This prevents self-overlap penalty which would force radius to 0
    np.fill_diagonal(dists, np.inf)
    
    # Sum of radii matrix
    sum_rs = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Violations matrix
    violations = np.maximum(0, sum_rs - dists)
    
    # Sum of squared violations (divide by 2 because matrix is symmetric)
    penalty += np.sum(violations**2) / 2.0
    
    return obj + weight * penalty

def run_packing():
    n = 26
    
    # 1. Initial Configuration: Hexagonal Grid
    # Generate a hexagonal lattice of points to serve as a good starting point
    # This provides a dense initial packing which helps the optimizer
    r_gen = 0.12
    centers = []
    y = r_gen
    row_idx = 0
    count = 0
    
    # We need n points
    while count < n:
        x = r_gen
        shift = r_gen if row_idx % 2 == 1 else 0
        x += shift
        
        # Add points in current row
        while x + r_gen <= 1.0 and count < n:
            centers.append([x, y])
            count += 1
            x += 2 * r_gen
        y += np.sqrt(3) * r_gen
        row_idx += 1
        
    centers = np.array(centers)
    
    # 2. Normalize to fit in [0.05, 0.95] to allow room for expansion
    if len(centers) > 0:
        xmin, ymin = centers.min(axis=0)
        xmax, ymax = centers.max(axis=0)
        w = xmax - xmin
        h = ymax - ymin
        
        target_w, target_h = 0.9, 0.9
        scale_x = target_w / w if w > 0 else 1
        scale_y = target_h / h if h > 0 else 1
        scale = min(scale_x, scale_y)
        
        centers = (centers - [xmin, ymin]) * scale + [0.05, 0.05]
    else:
        centers = np.random.rand(n, 2)
        
    # 3. Initial Radii
    # Estimate based on nearest neighbor distance to ensure valid start
    min_dist = 2.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
    
    # Start with radius slightly less than half the minimum distance
    r_init = min_dist / 2 * 0.95
    radii = np.full(n, r_init)
    
    # 4. Optimization
    # Flatten centers and radii into a single vector
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    # Run optimization using L-BFGS-B which supports bounds
    # args=(n,) passes n to the objective function
    res = minimize(objective, x0, args=(n,), method='L-BFGS-B', 
                   bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-12})
    
    best_centers = res.x[:2*n].reshape(n, 2)
    best_radii = res.x[2*n:]
    
    # 5. Post-processing for strict validity
    
    # Clip radii to satisfy boundary constraints strictly
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        # r must be <= x, <= 1-x, <= y, <= 1-y
        limit = min(x, 1-x, y, 1-y)
        if r > limit:
            best_radii[i] = max(0.0, limit)
            
    # Resolve overlaps by shrinking radii iteratively
    # This ensures no overlaps remain due to numerical precision
    for _ in range(50):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                sum_r = best_radii[i] + best_radii[j]
                if sum_r > dist + 1e-12:
                    diff = sum_r - dist
                    reduction = diff / 2
                    best_radii[i] = max(0.0, best_radii[i] - reduction)
                    best_radii[j] = max(0.0, best_radii[j] - reduction)
                    overlap_found = True
        if not overlap_found:
            break
            
    sum_radii = float(np.sum(best_radii))
    return best_centers, best_radii, sum_radii
