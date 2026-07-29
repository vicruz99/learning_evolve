# sol_000289 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 86cff419) state=0782d949 sum of radii=2.624863 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(vars, n, lam):
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    loss = -np.sum(radii)
    penalty = 0.0
    
    # Vectorized inter-circle penalty
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap = rad_sum - dists
    
    # Consider only pairs i < j
    overlap_upper = np.triu(overlap, k=1)
    pos_overlap = np.maximum(overlap_upper, 0.0)
    penalty += np.sum(pos_overlap**2)
    
    # Boundary penalty
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    penalty += np.sum(np.maximum(r - x, 0.0)**2)
    penalty += np.sum(np.maximum(x + r - 1.0, 0.0)**2)
    penalty += np.sum(np.maximum(r - y, 0.0)**2)
    penalty += np.sum(np.maximum(y + r - 1.0, 0.0)**2)
        
    return loss + lam * penalty

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.06)
    
    # Hexagonal initialization to promote dense packing
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx >= n:
                break
            x = 0.1 + j * 0.18
            y = 0.1 + i * 0.18
            if i % 2 == 1:
                x += 0.09
            centers[idx] = [x, y]
            idx += 1
            
    x0 = np.concatenate([centers.ravel(), radii])
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    current_vars = x0.copy()
    
    # Homotopy optimization: gradually increase penalty weight
    # This helps navigate the non-convex landscape and avoid getting stuck early
    for lam in [10, 50, 200, 1000, 5000, 20000, 100000]:
        res = minimize(compute_loss, current_vars, args=(n, lam), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-10})
        current_vars = res.x
        
    centers_opt = current_vars[:2*n].reshape(n, 2)
    radii_opt = current_vars[2*n:]
    
    # Strict constraint enforcement post-processing
    s = 1e-10
    # Inter-circle violations
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
            ov = radii_opt[i] + radii_opt[j] - d
            if ov / 2 > s:
                s = ov / 2
    # Boundary violations
    for i in range(n):
        x, y = centers_opt[i]
        r = radii_opt[i]
        ov = max(r - x, x + r - 1, r - y, y + r - 1, 0.0)
        if ov > s:
            s = ov
            
    if s > 0:
        radii_opt -= s
        radii_opt = np.maximum(radii_opt, 0.0)
        
    sum_r = np.sum(radii_opt)
    return centers_opt, radii_opt, sum_r
