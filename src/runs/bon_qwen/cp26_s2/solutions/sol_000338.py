# sol_000338 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d755ba05) state=e7748296 sum of radii=2.369034 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_overlap_penalty(centers_flat, r):
    """
    Computes the squared overlap penalty for a given configuration of centers and radius r.
    """
    centers = centers_flat.reshape(26, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Overlap occurs when distance < 2r
    overlaps = np.maximum(0.0, 2.0 * r - dists)
    
    # Ignore self-distances on the diagonal
    overlaps[np.diag_indices_from(overlaps)] = 0.0
    
    return np.sum(overlaps**2)

def run_packing():
    n = 26
    # Target radius corresponding to sum ~ 2.636
    target_r = 0.10135
    best_centers = None
    best_pen = np.inf
    
    np.random.seed(42)
    
    # Multiple restarts to avoid local minima
    for restart in range(12):
        # Base 5x5 grid initialization
        gx = np.linspace(0.1, 0.9, 5)
        gy = np.linspace(0.1, 0.9, 5)
        xs, ys = np.meshgrid(gx, gy)
        base = np.vstack([xs.ravel(), ys.ravel()]).T
        
        # Add small random perturbation
        pert = 0.01 * np.random.randn(n-1, 2)
        centers = base + pert
        
        # Place the 26th circle somewhere inside the square
        c26 = np.random.rand(1, 2) * 0.6 + 0.2
        centers = np.vstack([centers, c26])
        
        # Bound constraints: centers must be within [r, 1-r]
        low = target_r
        high = 1.0 - target_r
        bounds = [(low, high)] * (n * 2)
        
        res = minimize(compute_overlap_penalty, centers.flatten(), args=(target_r,),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
        
        if res.fun < best_pen:
            best_pen = res.fun
            best_centers = res.x.reshape(n, 2)
            
        # Early exit if a valid packing is found
        if best_pen < 1e-9:
            break
            
    if best_centers is None:
        return np.zeros((n, 2)), np.zeros(n), 0.0
        
    centers = best_centers
    
    # Compute the exact maximum feasible radius for this configuration
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    dists[np.diag_indices_from(dists)] = np.inf
    min_d = np.min(dists)
    
    # Distance to boundaries
    min_bx = np.min(centers[:, 0])
    max_bx = np.max(centers[:, 0])
    min_by = np.min(centers[:, 1])
    max_by = np.max(centers[:, 1])
    min_dist_bound = min(min_bx, 1.0 - max_bx, min_by, 1.0 - max_by)
    
    # Feasible radius is limited by closest pair and closest boundary
    r = min(min_d / 2.0, min_dist_bound)
    r = max(0.0, r)
    
    radii = np.full(n, r)
    return centers, radii, float(np.sum(radii))
