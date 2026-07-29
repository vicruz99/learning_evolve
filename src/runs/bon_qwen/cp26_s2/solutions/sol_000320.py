# sol_000320 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ef4a4e64) state=9be805fc sum of radii=2.538796 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(x, n=26, lam=1000.0):
    """
    Computes the negative sum of radii plus penalty terms for overlaps and boundary violations.
    """
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    sum_r = np.sum(radii)
    
    # Vectorized distance matrix computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Radius sums matrix
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Overlap penalty: strictly upper triangle to avoid double counting
    overlap = np.maximum(0, r_sum - dists)
    triu_indices = np.triu_indices(n, k=1)
    overlap_penalty = np.sum(overlap[triu_indices]**2)
    
    # Boundary penalty
    b_penalty = (np.sum(np.maximum(0, radii - centers[:, 0])**2) +
                 np.sum(np.maximum(0, centers[:, 0] + radii - 1)**2) +
                 np.sum(np.maximum(0, radii - centers[:, 1])**2) +
                 np.sum(np.maximum(0, centers[:, 1] + radii - 1)**2))
                 
    return -sum_r + lam * (overlap_penalty + b_penalty)

def run_packing():
    n = 26
    best_x = None
    best_obj = float('inf')
    
    # Base grid layout for initialization (5x5 covers 25 circles efficiently)
    grid_coords = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
    
    # Run multiple trials with different initial conditions to escape local minima
    for trial in range(6):
        if trial == 0:
            # Grid + one circle placed in a known gap
            init_centers = np.vstack([grid_coords, [0.25, 0.75]])
        else:
            # Randomized perturbation of the grid layout
            np.random.seed(42 + trial * 7)
            init_centers = grid_coords.copy()
            init_centers = np.vstack([init_centers, np.random.uniform(0, 1, size=(1, 2))])
            init_centers += np.random.uniform(-0.015, 0.015, size=(n, 2))
            
        x0_radii = np.full(n, 0.095)
        x0 = np.concatenate([init_centers.flatten(), x0_radii])
        
        bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
        
        try:
            res = minimize(compute_objective, x0, args=(n, 1500.0), method='L-BFGS-B', 
                           bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-12})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x.copy()
        except Exception:
            continue
            
    if best_x is None:
        # Fallback to simple grid if optimization fails
        centers = grid_coords
        centers = np.vstack([centers, [0.5, 0.5]])
        radii = np.full(n, 0.08)
        return centers, radii, np.sum(radii)

    centers = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]
    
    # Post-processing: ensure strict validity by checking clearances and scaling if necessary
    min_scale = 1.0
    for i in range(n):
        cx, cy = centers[i]
        r = radii[i]
        # Boundary clearances
        if r > cx: min_scale = min(min_scale, cx / r)
        if r > 1 - cx: min_scale = min(min_scale, (1 - cx) / r)
        if r > cy: min_scale = min(min_scale, cy / r)
        if r > 1 - cy: min_scale = min(min_scale, (1 - cy) / r)
        
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            req = radii[i] + radii[j]
            if req > dist:
                min_scale = min(min_scale, dist / req)
                
    # Apply scaling with a tiny margin to satisfy tolerance requirements strictly
    safe_scale = min_scale * 0.9995
    radii *= safe_scale
    
    return centers, radii, float(np.sum(radii))
