# sol_000159 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 39a6f529) state=a34118ef sum of radii=2.102440 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def objective(x, n):
    """
    Objective function to maximize the radius of equal circles.
    Minimizes negative of the bottleneck radius constraint.
    
    Args:
        x: Flattened array of centers (n * 2)
        n: Number of circles
    """
    c = x.reshape(n, 2)
    
    # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    # We track 2*r for consistency with pairwise distance constraint (2r <= dist)
    # 2r <= 2 * min(x, 1-x, y, 1-y)
    
    d_left = c[:, 0]
    d_right = 1.0 - c[:, 0]
    d_bottom = c[:, 1]
    d_top = 1.0 - c[:, 1]
    
    # Minimum distance to boundary for each circle
    min_bound_dist = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    global_min_bound = np.min(min_bound_dist)
    
    # Pairwise constraints: 2r <= distance(c_i, c_j)
    # Compute distance matrix efficiently using broadcasting
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    # Extract upper triangle distances (i < j) to avoid duplicates and self-distance
    mask = np.triu_indices(n, k=1)
    upper_tri_dists = dists[mask]
    global_min_pair = np.min(upper_tri_dists)
    
    # The feasible 2r is limited by the tightest constraint (boundary or neighbor)
    val = min(global_min_pair, 2.0 * global_min_bound)
    
    # Return negative for minimization (maximizing val)
    return -val

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal Cluster
    # Generate points on a triangular lattice to approximate hexagonal packing
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    points = []
    # Search range for lattice indices to generate enough points
    for i in range(-10, 11):
        for j in range(-10, 11):
            p = i * v1 + j * v2
            points.append(p)
    points = np.array(points)
    
    # Select 26 points closest to the origin to form a compact cluster
    dists_from_origin = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists_from_origin)[:n]
    centers = points[indices]
    
    # Center the cluster at (0.5, 0.5)
    centers = centers - np.mean(centers, axis=0) + 0.5
    
    # Scale to fit comfortably inside [0, 1]
    # Ensure max distance from center is <= 0.45 to leave margin for optimization
    max_dist = np.max(np.linalg.norm(centers - 0.5, axis=1))
    if max_dist > 0.45:
        centers = (centers - 0.5) * (0.45 / max_dist) + 0.5
        
    x0 = centers.flatten()
    
    # 2. Optimization
    # Use Nelder-Mead to find centers that maximize the minimum separation
    # This effectively maximizes the radius of equal circles
    try:
        res = opt.minimize(objective, x0, args=(n,), method='Nelder-Mead', 
                           options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 5000})
        optimal_centers = res.x.reshape(n, 2)
    except Exception:
        # Fallback to initial centers if optimization fails
        optimal_centers = x0.reshape(n, 2)
        
    # 3. Compute Final Radii
    c = optimal_centers
    
    # Boundary constraints
    d_left = c[:, 0]
    d_right = 1.0 - c[:, 0]
    d_bottom = c[:, 1]
    d_top = 1.0 - c[:, 1]
    min_bound = np.min(np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top)))
    
    # Pairwise constraints
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    mask = np.triu_indices(n, k=1)
    min_pair = np.min(dists[mask])
    
    # Radius is half the minimum pairwise distance, capped by boundary distance
    r = min(min_pair / 2.0, min_bound)
    
    # Apply small epsilon to ensure strict non-overlap within tolerance
    r = max(0.0, r - 1e-9)
    
    radii = np.full(n, r)
    sum_radii = np.sum(radii)
    
    return optimal_centers, radii, sum_radii
