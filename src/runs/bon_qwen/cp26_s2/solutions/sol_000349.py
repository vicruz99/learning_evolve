# sol_000349 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 03d022f0) state=b8fa38ba sum of radii=2.424373 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def get_min_separation(centers):
    """
    Calculates the maximum feasible radius 'r' for a given set of centers.
    Constraints:
    1. r <= min(x, y, 1-x, 1-y) for all centers (boundary constraint)
    2. r <= distance(i, j) / 2 for all pairs (non-overlap constraint)
    """
    n = centers.shape[0]
    if n == 0:
        return 0.0
    
    # Boundary constraints
    min_boundary = min(np.min(centers[:, 0]), np.min(centers[:, 1]),
                       np.min(1.0 - centers[:, 0]), np.min(1.0 - centers[:, 1]))
    
    # Inter-circle constraints
    # Calculate pairwise distances efficiently
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_inter = np.min(dists) / 2.0
    
    return min(min_boundary, min_inter)

def objective_function(x, n):
    """Objective function for minimization: we maximize min separation."""
    centers = x.reshape((n, 2))
    return -get_min_separation(centers)

def run_packing():
    n = 26
    
    # 1. Generate initial hexagonal lattice
    r_init = 0.102
    pts = []
    y = 0.0
    row = 0
    while len(pts) < n:
        x = 0.0
        shift = (row % 2) * r_init
        while x + 2 * r_init <= 1.0 + 1e-9:
            pts.append([x + r_init + shift, y + r_init])
            x += 2 * r_init
        row += 1
        y += r_init * math.sqrt(3)
    
    centers0 = np.array(pts[:n])
    centers0 = np.clip(centers0, 1e-6, 1.0 - 1e-6)
    
    best_r = get_min_separation(centers0)
    best_centers = centers0
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # 2. Optimization with multiple restarts
    np.random.seed(42)
    for i in range(10):
        if i == 0:
            x0 = centers0.flatten()
        else:
            # Perturb the best solution found so far
            noise = np.random.normal(0, 0.005, centers0.shape)
            x0 = best_centers + noise
            x0 = np.clip(x0, 0.001, 0.999)
            x0 = x0.flatten()
        
        try:
            res = minimize(objective_function, x0, args=(n,), 
                           method='SLSQP', bounds=bounds, 
                           options={'maxiter': 3000, 'ftol': 1e-12})
            c_new = res.x.reshape((n, 2))
            r_new = get_min_separation(c_new)
            if r_new > best_r:
                best_r = r_new
                best_centers = c_new
        except Exception:
            pass
            
    radii = np.full(n, best_r)
    total_sum = float(np.sum(radii))
    
    return best_centers, radii, total_sum
