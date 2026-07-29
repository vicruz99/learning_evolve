# sol_000124 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 22de7e34) state=20636cf0 sum of radii=2.296582 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n, lam):
    """
    Compute the penalized objective: -sum(radii) + penalty for constraint violations.
    vars: flattened array of [x1, y1, ..., xn, yn, r1, ..., rn]
    n: number of circles
    lam: penalty weight
    """
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Primary objective: maximize sum of radii (minimize negative sum)
    obj = -np.sum(r)
    
    penalty = 0.0
    
    # Boundary penalties: circles must be inside [0,1]x[0,1]
    for k in range(2):
        x = c[:, k]
        # r - x <= 0  => penalty if r > x
        over1 = r - x
        penalty += np.sum(np.maximum(0, over1)**2)
        # r - (1-x) <= 0 => penalty if r > 1-x
        over2 = r - (1.0 - x)
        penalty += np.sum(np.maximum(0, over2)**2)
        
    # Overlap penalties: dist(i,j) >= r_i + r_j
    for i in range(n):
        diffs = c[i:i+1] - c[i+1:]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        r_sum = r[i] + r[i+1:]
        over = r_sum - dists
        penalty += np.sum(np.maximum(0, over)**2)
        
    return obj + lam * penalty

def run_packing():
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    lam = 1000.0
    num_restarts = 3
    
    for restart in range(num_restarts):
        # Deterministic but varied initialization
        np.random.seed(restart * 100 + 7)
        centers = np.random.rand(n, 2) * 0.6 + 0.2
        radii = np.full(n, 0.04)
        
        x0 = np.concatenate([centers.flatten(), radii])
        bounds = [(0.0, 1.0) for _ in range(2*n)] + [(0.0, 0.5) for _ in range(n)]
        
        res = minimize(compute_objective, x0, args=(n, lam), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-10})
                       
        c_opt = res.x[:2*n].reshape(n, 2)
        r_opt = res.x[2*n:]
        
        curr_sum = np.sum(r_opt)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
            
    # Final adjustment to guarantee strict validity per validator tolerances
    for _ in range(20):
        valid = True
        # Check boundaries
        for i in range(n):
            x, y = best_centers[i]
            r = best_radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                valid = False
                break
        if valid:
            # Check overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                    if dist < best_radii[i] + best_radii[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            break
        # Slight shrinkage to resolve numerical drift
        best_radii *= 0.995
        
    best_radii = np.clip(best_radii, 0.0, None)
    best_sum = np.sum(best_radii)
    return best_centers, best_radii, best_sum
