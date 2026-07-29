# sol_000050 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000044 (state 69bc282d) state=d192014e sum of radii=2.389954 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 26

def compute_objective(centers_flat):
    """
    Computes the negative sum of maximum valid radii for a given configuration of centers.
    Vectorized for performance.
    """
    centers = centers_flat.reshape(N, 2)
    # Ensure centers stay strictly within bounds to prevent degenerate gradients
    centers = np.clip(centers, 1e-7, 1.0 - 1e-7)
    
    # Distance to boundaries: min(x, 1-x, y, 1-y)
    dist_bound = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                            np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Ignore self-distances by setting diagonal to infinity
    dists[np.arange(N), np.arange(N)] = np.inf
    
    # Radius limited by half the distance to the nearest neighbor
    dist_neighbor = 0.5 * np.min(dists, axis=1)
    
    # Valid radius is the tighter constraint between boundary and neighbors
    radii = np.minimum(dist_bound, dist_neighbor)
    return -np.sum(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize sum of radii."""
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Phase 1: Global exploration to locate high-density basin
    de_result = differential_evolution(
        compute_objective,
        bounds,
        seed=42,
        maxiter=150,
        popsize=25,
        mutation=(0.5, 1.2),
        recombination=0.8,
        tol=1e-9,
        polish=True
    )
    
    best_centers_flat = de_result.x
    
    # Phase 2: Local refinement for high precision
    local_result = minimize(
        compute_objective,
        best_centers_flat,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 15000, 'ftol': 1e-15, 'gtol': 1e-12}
    )
    
    centers = local_result.x.reshape(N, 2)
    
    # Phase 3: Exact radius computation for the final configuration
    dist_bound = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                            np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    dists[np.arange(N), np.arange(N)] = np.inf
    dist_neighbor = 0.5 * np.min(dists, axis=1)
    radii = np.minimum(dist_bound, dist_neighbor)
    
    # Ensure non-negativity to satisfy validator strictly
    radii = np.maximum(radii, 0.0)
    
    total_sum = float(np.sum(radii))
    return centers, radii, total_sum
