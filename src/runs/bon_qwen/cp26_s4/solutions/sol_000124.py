# sol_000124 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 40ff4175) state=f26c3e85 sum of radii=0.756242 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(x_vars, radii, n, triu_idx):
    """Compute squared penalty for boundary and overlap violations."""
    centers = x_vars.reshape((n, 2))
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii
    
    # Boundary penalties
    pen = np.sum(np.maximum(r - x, 0)**2)
    pen += np.sum(np.maximum(x + r - 1, 0)**2)
    pen += np.sum(np.maximum(r - y, 0)**2)
    pen += np.sum(np.maximum(y + r - 1, 0)**2)
    
    # Pairwise overlap penalties
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    min_d = r[:, np.newaxis] + r[np.newaxis, :]
    overlap = np.maximum(min_d - dist, 0)
    pen += np.sum(overlap[triu_idx]**2)
    return pen

def run_packing():
    n = 26
    triu_idx = np.triu_indices(n, k=1)
    
    # Initialize centers on a hexagonal grid pattern
    centers = np.zeros((n, 2))
    idx = 0
    cols = 6
    rows = 5
    for r_idx in range(rows):
        for c_idx in range(cols):
            if idx >= n:
                break
            x = (c_idx + 0.5 + (r_idx % 2) * 0.5) / cols
            y = (r_idx + 0.5) / rows
            centers[idx] = [x * 0.8 + 0.1, y * 0.8 + 0.1]
            idx += 1
            
    radii = np.full(n, 0.02)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    bounds = [(0.0, 1.0)] * (2 * n)
    max_outer = 150
    growth = 1.0025
    
    # Initial optimization to ensure valid start configuration
    x0 = centers.flatten()
    res = minimize(compute_penalty, x0, args=(radii, n, triu_idx), method='L-BFGS-B', 
                   bounds=bounds, options={'maxiter': 300, 'ftol': 1e-10})
    centers = res.x.reshape((n, 2))
    
    no_improve_count = 0
    
    for it in range(max_outer):
        x0 = centers.flatten()
        res = minimize(compute_penalty, x0, args=(radii, n, triu_idx), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 150, 'ftol': 1e-10})
        centers = res.x.reshape((n, 2))
        
        if res.fun < 1e-7:
            radii *= growth
            
            # Strict validation step
            valid = True
            s = 0.0
            for i in range(n):
                cx, cy = centers[i]
                cr = radii[i]
                s += cr
                if cx - cr < -1e-6 or cx + cr > 1 + 1e-6 or cy - cr < -1e-6 or cy + cr > 1 + 1e-6:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                        if d < radii[i] + radii[j] - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                        
            if valid and s > best_sum:
                best_sum = s
                best_centers = centers.copy()
                best_radii = radii.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            radii *= 0.995
            no_improve_count += 1
            
        if no_improve_count > 25:
            break
            
    return best_centers, best_radii, best_sum
