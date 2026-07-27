# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6773994b) state=cb3595a1 sum of radii=0.000127 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _compute_loss(params, n, lam):
    """Computes the objective: negative sum of radii + penalty for violations."""
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Vectorized pairwise distance and overlap calculation
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    rad_sum = radii[:, None] + radii[None, :]
    
    overlap = rad_sum - dist
    overlap = np.maximum(overlap, 0.0)
    p_overlap = np.sum(overlap**2)
    
    # Boundary penalty
    p_bound = 0.0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r: p_bound += (x - r)**2
        if x > 1.0 - r: p_bound += (x - (1.0 - r))**2
        if y < r: p_bound += (y - r)**2
        if y > 1.0 - r: p_bound += (y - (1.0 - r))**2
        
    return -np.sum(radii) + lam * (p_overlap + p_bound)

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.05
    
    # Initialize with a hexagonal-like pattern for better starting density
    k = 0
    for row in range(6):
        n_row = 4 if row == 0 or row == 5 else 5
        y = (row + 0.5) * 0.15
        for col in range(n_row):
            x = (col + (row % 2) * 0.5) * 0.15 + 0.075
            if k < n:
                centers[k] = [x, y]
                k += 1
    while k < n:
        centers[k] = [0.5, 0.5]
        k += 1
        
    # Add tiny fixed perturbation to break symmetry and aid gradient descent
    centers[:, 0] += 0.002 * np.arange(n)
    centers[:, 1] += 0.001 * np.arange(n)
    
    params = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0) for _ in range(2*n)] + [(0.0, 0.5) for _ in range(n)]
    
    best_params = params.copy()
    lam = 50.0
    
    # Iteratively increase penalty to enforce constraints strictly
    for _ in range(10):
        res = minimize(_compute_loss, best_params, args=(n, lam), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12})
        best_params = res.x
        lam *= 2.0
        
    centers_opt = best_params[:2*n].reshape(n, 2)
    radii_opt = best_params[2*n:]
    
    # Strict boundary enforcement
    for i in range(n):
        r = radii_opt[i]
        centers_opt[i, 0] = np.clip(centers_opt[i, 0], r, 1.0 - r)
        centers_opt[i, 1] = np.clip(centers_opt[i, 1], r, 1.0 - r)
        
    # Resolve any remaining microscopic overlaps by scaling radii down slightly if needed
    for _ in range(5):
        max_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers_opt[i, 0] - centers_opt[j, 0]
                dy = centers_opt[i, 1] - centers_opt[j, 1]
                d = np.sqrt(dx*dx + dy*dy)
                if d < radii_opt[i] + radii_opt[j]:
                    viol = (radii_opt[i] + radii_opt[j]) - d
                    if viol > max_viol:
                        max_viol = viol
        if max_viol > 1e-10:
            scale = 1.0 - max_viol / 2.0
            radii_opt *= scale
        else:
            break
            
    radii_opt = np.maximum(radii_opt, 0.0)
    return centers_opt, radii_opt, float(np.sum(radii_opt))
