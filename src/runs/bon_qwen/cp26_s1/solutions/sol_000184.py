# sol_000184 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9afef83a) state=d6f36619 sum of radii=2.143013 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_radius(centers):
    """Compute the maximum feasible equal radius for a given set of centers."""
    n = centers.shape[0]
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    # Mask out self-distances
    dists = np.where(np.eye(n), np.inf, dists)
    min_pair = np.min(dists) / 2.0
    
    # Distance to boundaries
    bound_dists = np.min(np.concatenate([centers, 1.0 - centers], axis=1), axis=1)
    min_bound = np.min(bound_dists)
    
    return min(min_pair, min_bound)

def objective(centers_flat):
    """Objective function to minimize (negative of max feasible radius)."""
    centers = centers_flat.reshape(-1, 2)
    return -get_radius(centers)

def run_packing():
    n = 26
    # Initialize with a 5x5 grid plus one center circle
    centers = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
    centers = np.vstack([centers, [0.5, 0.5]])
    
    # Add small random perturbation to break symmetry and escape local minima
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    # Optimize positions to maximize the minimum separation distance
    res = minimize(objective, centers.flatten(), method='Nelder-Mead',
                   options={'maxiter': 8000, 'xatol': 1e-5, 'fatol': 1e-7})
    
    best_centers = res.x.reshape(-1, 2)
    r = get_radius(best_centers)
    radii = np.full(n, r)
    
    return best_centers, radii, 26 * r
