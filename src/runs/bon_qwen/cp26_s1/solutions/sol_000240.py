# sol_000240 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5b918207) state=af282913 sum of radii=1.280314 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(x, n, P):
    centers = x[:2*n].reshape(n, 2)
    r = x[2*n]
    
    # Boundary penalties
    left = np.maximum(0.0, r - centers[:, 0])
    right = np.maximum(0.0, centers[:, 0] - (1.0 - r))
    bottom = np.maximum(0.0, r - centers[:, 1])
    top = np.maximum(0.0, centers[:, 1] - (1.0 - r))
    pen_bound = np.sum(left**2 + right**2 + bottom**2 + top**2)
    
    # Overlap penalties
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    violation = np.maximum(0.0, 2.0 * r - dists)
    pen_overlap = np.sum(violation**2)
    
    return -r + P * (pen_bound + pen_overlap)

def run_packing():
    n = 26
    P = 5000.0
    
    best_x = None
    best_loss = np.inf
    
    np.random.seed(42)
    for i in range(5):
        if i == 0:
            # Hexagonal-like initialization for a strong starting point
            centers_init = np.zeros((n, 2))
            r_init = 0.105
            idx = 0
            y = r_init
            shift = 0.0
            while idx < n:
                x = r_init + shift
                while x + r_init <= 1.0 and idx < n:
                    centers_init[idx, 0] = x
                    centers_init[idx, 1] = y
                    idx += 1
                    x += 2.0 * r_init
                y += np.sqrt(3) * r_init
                shift = r_init if shift == 0.0 else 0.0
        else:
            centers_init = np.random.uniform(0.15, 0.85, size=(n, 2))
            r_init = 0.10
            
        x0 = np.concatenate([centers_init.flatten(), [r_init]])
        bounds = [(0.0, 1.0)] * (2*n) + [(0.01, 0.20)]
        
        res = minimize(objective_func, x0, args=(n, P), method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 5000, 'ftol': 1e-12})
        
        if res.fun < best_loss:
            best_loss = res.fun
            best_x = res.x.copy()
            
    centers_opt = np.clip(best_x[:2*n].reshape(n, 2), 0.0, 1.0)
    
    # Compute exact feasible radius to guarantee validity
    diff = centers_opt[:, np.newaxis, :] - centers_opt[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_dist_left = np.min(centers_opt[:, 0])
    min_dist_right = np.min(1.0 - centers_opt[:, 0])
    min_dist_bottom = np.min(centers_opt[:, 1])
    min_dist_top = np.min(1.0 - centers_opt[:, 1])
    
    r_final = min(min_pair_dist / 2.0, min_dist_left, min_dist_right, min_dist_bottom, min_dist_top)
    r_final = max(r_final, 1e-7)
    
    radii = np.full(n, r_final)
    sum_radii = np.sum(radii)
    
    return centers_opt, radii, sum_radii
