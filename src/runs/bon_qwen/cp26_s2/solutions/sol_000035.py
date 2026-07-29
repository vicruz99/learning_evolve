# sol_000035 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 16d9c155) state=0a555ba6 sum of radii=2.200000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _compute_objective(centers_flat):
    """
    Objective function for scipy.optimize.minimize.
    Returns the negative sum of radii to convert maximization to minimization.
    """
    centers = centers_flat.reshape(26, 2)
    
    # 1. Distance to boundaries
    left_dist = centers[:, 0]
    right_dist = 1.0 - centers[:, 0]
    bottom_dist = centers[:, 1]
    top_dist = 1.0 - centers[:, 1]
    
    bound_dist = np.minimum(np.minimum(left_dist, right_dist), 
                            np.minimum(bottom_dist, top_dist))
    
    # 2. Distance to other circles
    # Vectorized pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)  # Ignore self-distance
    min_neighbor_dist = np.min(dists, axis=1)
    
    # 3. Radii are limited by both boundaries and neighbors
    radii = np.minimum(bound_dist, min_neighbor_dist * 0.5)
    
    # Return negative sum for minimization
    return -np.sum(radii)


def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    
    # Initial configuration: 5x5 grid + 1 extra circle
    grid_x = np.linspace(0.15, 0.85, 5)
    grid_y = np.linspace(0.15, 0.85, 5)
    cx, cy = np.meshgrid(grid_x, grid_y)
    centers_init = np.vstack([cx.ravel(), cy.ravel()]).T
    centers_init = np.vstack([centers_init, [0.5, 0.05]])  # 26th circle
    
    bounds = [(0.0, 1.0)] * 52
    
    best_centers = centers_init.copy()
    best_sum_radii = -_compute_objective(centers_init.flatten())
    
    # Primary optimization
    res = minimize(_compute_objective, centers_init.flatten(), 
                   method='Powell', bounds=bounds, 
                   options={'maxiter': 6000, 'ftol': 1e-11})
    if -res.fun > best_sum_radii:
        best_centers = res.x.reshape(26, 2)
        best_sum_radii = -res.fun
        
    # Secondary runs with perturbations to escape local minima
    for _ in range(6):
        pert = centers_init.copy()
        pert += np.random.uniform(-0.04, 0.04, pert.shape)
        pert = np.clip(pert, 0.02, 0.98)
        
        res2 = minimize(_compute_objective, pert.flatten(), 
                        method='Powell', bounds=bounds, 
                        options={'maxiter': 4000, 'ftol': 1e-11})
        if -res2.fun > best_sum_radii:
            best_centers = res2.x.reshape(26, 2)
            best_sum_radii = -res2.fun
            
    # Final precise radius calculation
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_nd = np.min(dists, axis=1)
    
    bd = np.minimum(np.minimum(best_centers[:, 0], 1.0 - best_centers[:, 0]),
                    np.minimum(best_centers[:, 1], 1.0 - best_centers[:, 1]))
    radii = np.minimum(bd, min_nd * 0.5)
    
    return best_centers, radii, np.sum(radii)
