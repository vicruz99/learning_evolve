# sol_000005 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1057391e) state=f8c4b1bf sum of radii=1.930314 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import differential_evolution, minimize

def compute_sum_radii(centers):
    """Compute the maximum possible sum of radii for a given set of centers."""
    n = centers.shape[0]
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists, axis=1)
    
    # Distance to square boundaries
    min_bound_dist = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # Radii are limited by the nearest constraint
    radii = np.minimum(min_pair_dist, min_bound_dist) / 2.0
    return np.sum(radii)

def objective_function(centers_flat):
    """Objective to minimize: negative sum of radii + boundary penalty."""
    centers = centers_flat.reshape(26, 2)
    
    # Soft penalty to keep centers within [0, 1]
    out_lower = np.sum(np.maximum(0.0, -centers))
    out_upper = np.sum(np.maximum(0.0, centers - 1.0))
    boundary_penalty = out_lower + out_upper
    if boundary_penalty > 1e-12:
        return 1e4 * boundary_penalty
        
    return -compute_sum_radii(centers)

def run_packing():
    bounds = [(0.0, 1.0)] * 52
    
    # 1. Global search to find a good basin of attraction
    de_result = differential_evolution(
        objective_function,
        bounds,
        seed=42,
        maxiter=300,
        popsize=20,
        tol=1e-9,
        polish=False
    )
    
    # 2. Local refinement for precision
    loc_result = minimize(
        objective_function,
        de_result.x,
        method='Nelder-Mead',
        options={'maxiter': 8000, 'xatol': 1e-10, 'fatol': 1e-11}
    )
    
    best_centers = loc_result.x.reshape(26, 2)
    
    # 3. Exact radius computation from optimized centers
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists, axis=1)
    
    min_bound_dist = np.minimum(
        np.minimum(best_centers[:, 0], 1.0 - best_centers[:, 0]),
        np.minimum(best_centers[:, 1], 1.0 - best_centers[:, 1])
    )
    
    best_radii = np.minimum(min_pair_dist, min_bound_dist) / 2.0
    best_radii = np.maximum(best_radii, 0.0)
    
    # 4. Safety adjustments for validation tolerance
    best_centers = np.clip(best_centers, 1e-12, 1.0 - 1e-12)
    best_radii = np.maximum(best_radii - 1e-13, 0.0)
    
    total_sum = np.sum(best_radii)
    return best_centers, best_radii, total_sum
