# sol_000103 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bf51a1cd) state=e76a4f5d sum of radii=2.197401 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_function(params, n):
    """
    Objective function to maximize the minimum radius r.
    params: 1D array of size 2*n (x1, y1, ..., xn, yn)
    n: number of circles
    
    The function returns -min_r, so minimizing this function maximizes min_r.
    """
    centers = params.reshape(n, 2)
    
    # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    # This is equivalent to r <= min(x, 1-x, y, 1-y)
    xs = centers[:, 0]
    ys = centers[:, 1]
    # Distance to the nearest boundary
    # If a center is outside [0,1], this distance becomes negative.
    dist_to_bound = np.minimum(np.minimum(xs, 1.0 - xs), np.minimum(ys, 1.0 - ys))
    min_r_bound = np.min(dist_to_bound)
    
    # Pairwise constraints: 2r <= distance between centers => r <= distance / 2
    # Compute pairwise distances efficiently
    # (N, 1, 2) - (1, N, 2) -> (N, N, 2) difference matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Consider only unique pairs (i < j) to avoid self-distance and double counting
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pairwise_dists = dists[mask]
    min_pairwise_r = np.min(pairwise_dists) / 2.0
    
    # The maximum possible radius for this configuration is limited by the tightest constraint
    min_r = min(min_r_bound, min_pairwise_r)
    
    # We want to maximize min_r, so we minimize -min_r
    return -min_r

def run_packing():
    n = 26
    
    best_min_r = 0.0
    best_centers = None
    
    # Generate multiple starting configurations to avoid local minima
    starts = []
    
    # 1. Grid based start (5x5 + 1)
    # A 5x5 grid fits 25 circles of radius 0.1 exactly.
    # We add a 26th circle and let the optimizer adjust.
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    gx, gy = np.meshgrid(grid_x, grid_y)
    base_pts = np.vstack([gx.flatten(), gy.flatten()]).T
    # Add a point in a gap, e.g., near (0.2, 0.2)
    starts.append(np.vstack([base_pts, [0.2, 0.2]]).flatten())
    
    # 2. Random starts
    rng = np.random.default_rng(42)
    # Run several random restarts to explore the space
    for _ in range(30):
        starts.append(rng.uniform(0.1, 0.9, size=(n, 2)).flatten())

    # Optimize for each start configuration
    for p0 in starts:
        try:
            # Nelder-Mead is suitable for non-smooth objectives (min function)
            # maxiter is set high enough to converge
            res = minimize(objective_function, p0, args=(n,), method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            current_r = -res.fun
            # We are interested in valid packings where radius is positive
            # However, the optimizer might explore invalid regions (negative r).
            # We track the best positive radius found.
            if current_r > best_min_r:
                best_min_r = current_r
                best_centers = res.x.reshape(n, 2)
        except Exception:
            pass
            
    # Fallback if optimization failed completely
    if best_centers is None:
        best_centers = np.random.rand(n, 2)
        best_min_r = 0.0
        
    radii = np.full(n, best_min_r)
    
    # Safety clamp: ensure radius is non-negative
    if best_min_r < 0:
        best_min_r = 0.0
        radii = np.zeros(n)
        # Clamp centers to unit square just in case
        best_centers = np.clip(best_centers, 0.0, 1.0)
        
    return best_centers, radii, np.sum(radii)
