# sol_000038 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 5778b268) state=1f98ece2 sum of radii=2.463938 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(x_flat, mid, n):
    """
    Computes the squared penalty for boundary violations and circle overlaps
    for a fixed radius 'mid'.
    """
    c = x_flat.reshape(n, 2)
    pen = 0.0
    
    # Boundary violations: circles must be within [mid, 1-mid]
    vx = np.minimum(c[:,0] - mid, 1.0 - c[:,0] - mid)
    vy = np.minimum(c[:,1] - mid, 1.0 - c[:,1] - mid)
    pen += np.sum(np.maximum(0.0, -vx)**2)
    pen += np.sum(np.maximum(0.0, -vy)**2)
    
    # Pairwise overlap violations: distance must be >= 2*mid
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    viol = np.maximum(0.0, 2.0*mid - dists)
    pen += np.sum(viol**2)
    
    return pen

def run_packing():
    n = 26
    np.random.seed(42)
    
    # 1. Initialization: Hexagonal lattice pattern
    centers = np.zeros((n, 2))
    idx = 0
    r_init = 0.07
    y = r_init
    row = 0
    while idx < n:
        shift = r_init if row % 2 == 1 else 0.0
        x = r_init + shift
        while x + r_init <= 1.0 and idx < n:
            centers[idx] = [x, y]
            idx += 1
            x += 2 * r_init
        y += r_init * np.sqrt(3)
        row += 1
        
    # Fallback fill if hex pattern didn't yield exactly n points
    if idx < n:
        for i in range(idx, n):
            centers[i] = [0.5, 0.5]

    best_centers = centers.copy()
    best_r = 0.07
    
    # 2. Bisection / Expansion loop to find maximum feasible radius
    low = 0.07
    high = 0.12
    
    for _ in range(30):
        if high - low < 1e-6:
            break
        mid = (low + high) / 2.0
        
        found_valid = False
        
        # Try optimization from current best configuration
        res = minimize(compute_penalty, best_centers.flatten(), args=(mid, n), 
                       method='L-BFGS-B', options={'maxiter': 2000, 'ftol': 1e-15})
        
        if res.fun < 1e-5:
            best_centers = res.x.reshape(n, 2)
            best_r = mid
            low = mid
            found_valid = True
            
        # If stuck in local minimum, try random perturbations
        if not found_valid:
            for _ in range(3):
                perturb = best_centers + np.random.uniform(-0.01, 0.01, (n, 2))
                perturb = np.clip(perturb, 0.05, 0.95)
                res = minimize(compute_penalty, perturb.flatten(), args=(mid, n),
                               method='L-BFGS-B', options={'maxiter': 1500, 'ftol': 1e-15})
                if res.fun < 1e-5:
                    best_centers = res.x.reshape(n, 2)
                    best_r = mid
                    low = mid
                    found_valid = True
                    break
                    
        if not found_valid:
            high = mid

    # 3. Final Validation and Safety Scaling
    # Ensure strict compliance with validate_packing tolerances
    min_clearance = 1.0
    for i in range(n):
        min_clearance = min(min_clearance, best_centers[i,0], 1-best_centers[i,0], 
                            best_centers[i,1], 1-best_centers[i,1])
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            min_clearance = min(min_clearance, d - 2*best_r)
            
    # Shrink radii slightly if numerical precision caused marginal violations
    if min_clearance < -1e-7:
        scale = 0.999
        for _ in range(100):
            r_trial = best_r * scale
            valid = True
            for i in range(n):
                if best_centers[i,0] < r_trial or best_centers[i,0] > 1-r_trial or \
                   best_centers[i,1] < r_trial or best_centers[i,1] > 1-r_trial:
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        if np.linalg.norm(best_centers[i]-best_centers[j]) < 2*r_trial - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                best_r *= scale
                break
            scale *= 0.999

    radii = np.full(n, best_r)
    return best_centers, radii, float(np.sum(radii))
