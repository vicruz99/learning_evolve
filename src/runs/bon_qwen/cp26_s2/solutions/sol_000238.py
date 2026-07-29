# sol_000238 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e58a758a) state=4a82a837 sum of radii=2.138319 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_min_radius(centers):
    """Computes the maximum feasible equal radius for a given configuration of centers."""
    n = centers.shape[0]
    
    # Distance to boundaries
    d_left = centers[:, 0]
    d_right = 1.0 - centers[:, 0]
    d_bottom = centers[:, 1]
    d_top = 1.0 - centers[:, 1]
    min_boundary = np.min(np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top)))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair)

def packing_objective(x, n):
    """Objective function for optimization: negative of feasible radius."""
    return -compute_min_radius(x.reshape(n, 2))

def run_packing():
    n = 26
    best_r = 0.0
    best_centers = None
    
    # Strategy 1: Hexagonal lattice initialization
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(5):
        for col in range(6):
            if idx < n:
                offset = 0.5 / 6.0 if row % 2 == 1 else 0.0
                centers[idx, 0] = (col + 0.5) / 6.0 + offset
                centers[idx, 1] = (row + 0.5) / 5.0
                idx += 1
                
    # Optimize from hexagonal start
    res = minimize(packing_objective, centers.flatten(), args=(n,), method='Nelder-Mead',
                   options={'maxiter': 50000, 'xatol': 1e-7, 'fatol': 1e-9})
    curr_r = -res.fun
    if curr_r > best_r:
        best_r = curr_r
        best_centers = res.x.reshape(n, 2)
        
    # Strategy 2: Perturbed restarts to escape local minima
    for _ in range(5):
        if best_centers is None:
            break
        perturbed = best_centers.copy()
        perturbed += np.random.randn(n, 2) * 0.015
        perturbed = np.clip(perturbed, 0.02, 0.98)
        
        res = minimize(packing_objective, perturbed.flatten(), args=(n,), method='Nelder-Mead',
                       options={'maxiter': 30000, 'xatol': 1e-7, 'fatol': 1e-9})
        curr_r = -res.fun
        if curr_r > best_r:
            best_r = curr_r
            best_centers = res.x.reshape(n, 2)
            
    # Compute precise feasible radius and apply safety margin for validator tolerances
    final_r = compute_min_radius(best_centers) - 1e-11
    radii = np.full(n, final_r)
    
    return best_centers, radii, np.sum(radii)
