# sol_000096 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e1ebaf70) state=ca69f8fd sum of radii=2.631094 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def constraints_fast(vars_flat):
    """Compute boundary and overlap constraints for the packing."""
    centers = vars_flat[:52].reshape(26, 2)
    radii = vars_flat[52:]
    
    # Boundary constraints: 4 per circle (x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0)
    b = np.empty(104)
    for i in range(26):
        b[4*i]   = centers[i, 0] - radii[i]
        b[4*i+1] = 1.0 - centers[i, 0] - radii[i]
        b[4*i+2] = centers[i, 1] - radii[i]
        b[4*i+3] = 1.0 - centers[i, 1] - radii[i]
        
    # Overlap constraints: distance >= sum of radii for all pairs i < j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.tril(np.ones((26, 26), dtype=bool), k=-1)
    overlaps = dists[mask] - r_sum[mask]
    
    return np.concatenate([b, overlaps])

def neg_sum_radii(vars_flat):
    """Objective function: negative sum of radii (to be minimized)."""
    return -np.sum(vars_flat[52:])

def run_packing():
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Base grid initialization
    base_centers = np.zeros((26, 2))
    for i in range(26):
        base_centers[i, 0] = (i % 5 + 1) / 6.0
        base_centers[i, 1] = (i // 5 + 1) / 6.0
    base_radii = np.full(26, 0.06)
    
    bounds = [(0.0, 1.0)] * 52 + [(0.0, 0.5)] * 26
    cons = {'type': 'ineq', 'fun': constraints_fast}
    
    # Run multiple optimizations with different perturbations
    for seed in range(5):
        rng = np.random.default_rng(seed)
        centers = base_centers.copy()
        centers += rng.uniform(-0.03, 0.03, size=centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        radii = base_radii.copy()
        radii += rng.uniform(-0.01, 0.01, size=radii.shape)
        radii = np.clip(radii, 0.01, 0.2)
        
        x0 = np.concatenate([centers.flatten(), radii])
        
        res = minimize(
            fun=neg_sum_radii,
            x0=x0,
            bounds=bounds,
            constraints=cons,
            method='SLSQP',
            options={'maxiter': 400, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.success:
            curr_centers = res.x[:52].reshape(26, 2)
            curr_radii = res.x[52:]
            curr_sum = np.sum(curr_radii)
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = curr_centers.copy()
                best_radii = curr_radii.copy()
                
    # Fallback feasibility check & projection
    if best_centers is not None:
        scale = 1.0
        while scale > 0.01:
            r_test = best_radii * scale
            valid = True
            
            # Check boundaries
            for i in range(26):
                x, y = best_centers[i]
                r = r_test[i]
                if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                    valid = False
                    break
            if not valid:
                scale -= 0.02
                continue
                
            # Check overlaps
            for i in range(26):
                for j in range(i + 1, 26):
                    d = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                    if d < r_test[i] + r_test[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
                    
            if valid:
                best_radii = r_test
                best_sum = np.sum(best_radii)
                break
            scale -= 0.02
            
    return best_centers, best_radii, best_sum
