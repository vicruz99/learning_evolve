# sol_000084 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9821b492) state=a439dc32 sum of radii=2.158579 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(vars, n):
    centers = vars[:2*n].reshape((n, 2))
    radii = vars[2*n:]
    
    penalty = 0.0
    
    # Boundary constraints: circles must stay inside [0,1]x[0,1]
    penalty += np.sum(np.maximum(0, radii - centers[:, 0])**2)
    penalty += np.sum(np.maximum(0, radii - (1.0 - centers[:, 0]))**2)
    penalty += np.sum(np.maximum(0, radii - centers[:, 1])**2)
    penalty += np.sum(np.maximum(0, radii - (1.0 - centers[:, 1]))**2)
    
    # Overlap constraints: distance between centers >= sum of radii
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap = rad_sum - dists
    np.fill_diagonal(overlap, -np.inf)
    penalty += np.sum(np.maximum(0, overlap)**2)
    
    return penalty

def objective(vars, n):
    # Maximize sum of radii -> minimize negative sum
    # Penalty weight 5000 ensures constraints are tightly enforced
    return -np.sum(vars[2*n:]) + 5000.0 * compute_penalty(vars, n)

def run_packing():
    n = 26
    best_vars = None
    best_val = np.inf
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # Multiple restarts to avoid local minima
    for seed in range(20):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.05, 0.95, n)
        y = rng.uniform(0.05, 0.95, n)
        r = 0.09 + rng.uniform(0, 0.02, n)
        x0 = np.concatenate([x, y, r])
        
        res = minimize(objective, x0, args=(n,), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.fun < best_val:
            best_val = res.fun
            best_vars = res.x.copy()
            
    centers = best_vars[:2*n].reshape((n, 2))
    radii = best_vars[2*n:]
    
    # Strict projection to boundary constraints
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1.0-x, y, 1.0-y)
        
    # Iterative overlap resolution to guarantee validity
    for _ in range(50):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = rad_sum - dists
        np.fill_diagonal(overlap, 0)
        overlaps = np.maximum(0, overlap)
        if np.max(overlaps) < 1e-12:
            break
        for i in range(n):
            for j in range(i+1, n):
                if overlaps[i, j] > 1e-12:
                    shrink = overlaps[i, j] / 2.0
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    
    # Final boundary safety clamp
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1.0-x, y, 1.0-y)
        
    return centers, radii, float(np.sum(radii))
