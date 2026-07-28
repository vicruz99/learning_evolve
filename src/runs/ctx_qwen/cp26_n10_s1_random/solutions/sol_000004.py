# sol_000004 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4fe936d0) state=9d02857a sum of radii=1.699372 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def smooth_min_clearance(centers, k=40):
    """
    Computes a smooth approximation of the minimum clearance (to boundaries and other circles).
    k controls the sharpness of the approximation.
    """
    n = centers.shape[0]
    # Distance to each boundary
    b_clr = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    b_clr = np.minimum(b_clr, centers[:, 1])
    b_clr = np.minimum(b_clr, 1.0 - centers[:, 1])
    
    # Pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Constraints: boundary distance and half pairwise distance
    vals = np.concatenate([b_clr, dists[dists < np.inf] / 2.0])
    
    # Smooth min approximation: -1/k * log(sum(exp(-k*x)))
    return -np.log(np.sum(np.exp(-k * vals))) / k

def run_packing():
    n = 26
    k = 40
    
    # 1. Initialize in a hexagonal pattern
    centers = np.zeros((n, 2))
    r_init = 0.1
    dx = 2 * r_init
    dy = math.sqrt(3) * r_init
    idx = 0
    for r in range(5):
        for c in range(6):
            if idx >= n: break
            x = 0.05 + c * dx + (r % 2) * (dx / 2)
            y = 0.05 + r * dy
            centers[idx] = [x, y]
            idx += 1
            
    # Scale to fill [0,1] region effectively
    centers = (centers - centers.min(axis=0)) / (centers.max(axis=0) - centers.min(axis=0))
    centers = centers * 0.9 + 0.05
    
    bounds = [(0.0, 1.0) for _ in range(2 * n)]
    
    def objective(params):
        c = params.reshape(n, 2)
        return -smooth_min_clearance(c, k)
        
    # 2. Optimize positions to maximize clearance
    res = minimize(objective, centers.flatten(), method='L-BFGS-B', 
                   bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-14})
    
    final_centers = res.x.reshape(n, 2)
    
    # 3. Compute exact maximal radii for the optimized positions
    radii = np.zeros(n)
    for i in range(n):
        min_d = min(final_centers[i,0], 1-final_centers[i,0], 
                    final_centers[i,1], 1-final_centers[i,1])
        for j in range(n):
            if i != j:
                d = np.linalg.norm(final_centers[i] - final_centers[j])
                if d < min_d:
                    min_d = d
        radii[i] = min_d / 2.0
        
    sum_r = np.sum(radii)
    return final_centers, radii, sum_r
