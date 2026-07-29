# sol_000040 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 16d9c155) state=7d7504a4 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import differential_evolution

def compute_negative_sum_radii(x):
    """Objective function: returns the negative sum of feasible radii."""
    n = 26
    centers = x.reshape(n, 2)
    
    # Compute pairwise Euclidean distances
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, 0.0)
    
    radii = np.zeros(n)
    for i in range(n):
        # Distance to the four boundaries
        d_bound = min(
            centers[i, 0], 
            1.0 - centers[i, 0], 
            centers[i, 1], 
            1.0 - centers[i, 1]
        )
        # Half distance to the nearest neighbor
        d_neighbor = dists[i].min()
        
        # Feasible radius is limited by boundary and nearest circle
        radii[i] = min(d_bound, d_neighbor / 2.0)
        
    # We minimize the negative sum, equivalent to maximizing the sum
    return -np.sum(radii)

def run_packing():
    n = 26
    # Bounds for each coordinate: circles must stay within [0, 1]
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Global optimization using Differential Evolution
    # popsize=30 and maxiter=400 provide a good balance of exploration and speed
    result = differential_evolution(
        compute_negative_sum_radii, 
        bounds, 
        popsize=30, 
        maxiter=400, 
        tol=1e-9, 
        seed=42, 
        polish=False  # Disable polish to avoid issues with non-smooth objective landscape
    )
    
    centers = result.x.reshape(n, 2)
    
    # Final precise calculation of radii for the optimized centers
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, 0.0)
    radii = np.array([
        min(
            centers[i, 0], 
            1.0 - centers[i, 0], 
            centers[i, 1], 
            1.0 - centers[i, 1], 
            dists[i].min() / 2.0
        )
        for i in range(n)
    ])
    
    return centers, radii, np.sum(radii)
