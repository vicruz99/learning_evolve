# sol_000339 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6a87b209) state=13178fff sum of radii=1.960571 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _compute_obj(params, n, idx_i, idx_j, penalty_weight):
    """
    Compute the objective function: negative sum of radii + penalty for constraint violations.
    """
    x = params[:n]
    y = params[n:2*n]
    r = params[2*n:]
    
    # Boundary penalties: ensure circles are inside [0,1]x[0,1]
    pen = np.sum(np.maximum(0.0, r - x)**2 + np.maximum(0.0, r - (1.0 - x))**2)
    pen += np.sum(np.maximum(0.0, r - y)**2 + np.maximum(0.0, r - (1.0 - y))**2)
    
    # Overlap penalties: ensure dist(i,j) >= r_i + r_j
    c = np.column_stack((x, y))
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    d_ij = dist[idx_i, idx_j]
    r_ij = r[idx_i] + r[idx_j]
    
    pen += np.sum(np.maximum(0.0, r_ij - d_ij)**2)
    
    return -np.sum(r) + penalty_weight * pen

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    # Precompute indices for pairwise constraints (lower triangle)
    idx_i, idx_j = np.tril_indices(n, -1)
    penalty_weight = 10000.0
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Try multiple initializations to find a robust global optimum
    for seed in range(5):
        np.random.seed(seed)
        
        # Initialize centers on a perturbed 5x5 grid + 1 center point
        grid = np.linspace(0.1, 0.9, 5)
        x_grid = np.repeat(grid, 5)
        y_grid = np.tile(grid, 5)
        centers_init = np.column_stack((x_grid, y_grid))
        
        # Add 26th circle at center with slight perturbation
        centers_init = np.vstack([centers_init, [0.5, 0.5]])
        
        # Add random noise to break symmetry and avoid flat regions
        centers_init += np.random.uniform(-0.02, 0.02, centers_init.shape)
        centers_init = np.clip(centers_init, 0.02, 0.98)
        
        # Initial radii
        r_init = np.full(n, 0.09) + np.random.uniform(-0.005, 0.005, n)
        r_init = np.maximum(r_init, 0.01)
        
        params = np.concatenate([centers_init.ravel(), r_init])
        bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
        
        # Optimize
        res = minimize(_compute_obj, params, args=(n, idx_i, idx_j, penalty_weight),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-10, 'gtol': 1e-8})
        
        x, y, r = res.x[:n], res.x[n:2*n], res.x[2*n:]
        centers_opt = np.column_stack((x, y))
        
        # Post-process: Strictly enforce constraints
        # Iteratively shrink radii if overlaps exist
        for _ in range(100):
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(centers_opt[i] - centers_opt[j])
                    req = r[i] + r[j]
                    if d < req - 1e-12:
                        scale = (d + 1e-12) / req
                        r[i] *= np.sqrt(scale)
                        r[j] *= np.sqrt(scale)
                        changed = True
            
            # Enforce boundary constraints
            r = np.minimum(r, np.minimum(centers_opt[:, 0], 1.0 - centers_opt[:, 0]))
            r = np.minimum(r, np.minimum(centers_opt[:, 1], 1.0 - centers_opt[:, 1]))
            r = np.maximum(r, 0.0)
            
            if not changed:
                break
                
        current_sum = np.sum(r)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers_opt.copy()
            best_radii = r.copy()
            
    return best_centers, best_radii, best_sum
