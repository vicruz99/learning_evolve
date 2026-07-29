# sol_000129 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cd61366d) state=9d4a1bce sum of radii=2.452319 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(x, n):
    """
    Computes the negative sum of maximal radii for given center positions.
    x: flattened array of shape (2*n,) containing (x1, y1, x2, y2, ...)
    n: number of circles
    """
    centers = x.reshape(n, 2)
    
    # Compute pairwise Euclidean distances
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    # Compute distance to square boundaries
    bd_dists = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # Maximum valid radius is limited by boundary proximity and half nearest-neighbor distance
    radii = np.minimum(bd_dists, 0.5 * min_dists)
    
    # Return negative sum for minimization routines
    return -np.sum(radii)

def run_packing():
    n = 26
    best_val = np.inf
    best_x = None
    
    # Multiple restarts from structured initial guesses to avoid local minima
    for seed in range(10):
        rng = np.random.RandomState(seed)
        centers = np.zeros((n, 2))
        idx = 0
        row = 0
        
        # Initialize with a hexagonal-like pattern
        while idx < n:
            col = 0
            offset = 0.1 if row % 2 else 0.0
            while col < 7 and idx < n:
                x_pos = 0.1 + col * 0.18 + offset
                y_pos = 0.1 + row * 0.16
                if 0 <= x_pos <= 1 and 0 <= y_pos <= 1:
                    centers[idx] = [x_pos, y_pos]
                    idx += 1
                col += 1
            row += 1
            
        # Add controlled random perturbation
        noise = rng.uniform(-0.02, 0.02, size=centers.shape)
        centers = np.clip(centers + noise, 0.01, 0.99)
        
        # Optimize center positions
        res = minimize(objective_func, centers.ravel(), args=(n,), method='L-BFGS-B', 
                       bounds=[(0.0, 1.0)] * (2 * n), 
                       options={'maxiter': 5000, 'ftol': 1e-12})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    # Extract optimal centers
    best_centers = best_x.reshape(n, 2)
    
    # Recompute exact maximal radii for the final configuration
    dists = np.linalg.norm(best_centers[:, np.newaxis] - best_centers[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    bd_dists = np.minimum(
        np.minimum(best_centers[:, 0], 1.0 - best_centers[:, 0]),
        np.minimum(best_centers[:, 1], 1.0 - best_centers[:, 1])
    )
    
    best_radii = np.minimum(bd_dists, 0.5 * min_dists)
    
    # Apply microscopic shrinkage to guarantee strict validity within numerical tolerance
    best_radii = np.maximum(best_radii - 1e-9, 0.0)
    
    return best_centers, best_radii, np.sum(best_radii)
