# sol_000011 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state accdaaf6) state=38d2cf65 sum of radii=2.333597 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(params, n=26):
    r = params[0]
    centers = params[1:].reshape(n, 2)
    pen = 0.0
    
    # Boundary penalties
    x = centers[:, 0]
    y = centers[:, 1]
    pen += np.sum(np.maximum(0.0, r - x)**2)
    pen += np.sum(np.maximum(0.0, x - (1.0 - r))**2)
    pen += np.sum(np.maximum(0.0, r - y)**2)
    pen += np.sum(np.maximum(0.0, y - (1.0 - r))**2)
    
    # Overlap penalties (vectorized)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 2.0)  # Ignore self-distances
    
    violations = 2.0 * r - dists
    pos_viol = violations[violations > 0.0]
    pen += np.sum(pos_viol**2)
    
    return pen

def objective(params, mu, n=26):
    return -params[0] + mu * compute_penalty(params, n)

def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Hexagonal initialization
    rows = [6, 5, 6, 5, 4]  # Sum = 26
    r_init = 0.08
    centers = []
    y = 0.1
    dy = r_init * np.sqrt(3) * 1.1
    for idx, cnt in enumerate(rows):
        offset = dy / 2.0 if idx % 2 == 1 else 0.0
        x = 0.1 + offset
        dx = r_init * 2.0 * 1.1
        for _ in range(cnt):
            centers.append([x, y])
            x += dx
        y += dy
        
    centers = np.array(centers)
    centers = np.clip(centers, r_init, 1.0 - r_init)
    params = np.concatenate([[r_init], centers.flatten()])
    
    bounds = [(0.01, 0.5)] + [(0.0, 1.0)] * (2 * n)
    mu = 200.0
    
    # 2 & 3 & 4. Progressive optimization with perturbation
    for step in range(100):
        res = minimize(objective, params, args=(mu, n), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-9})
        params = res.x
        mu *= 1.25
        
        # Perturb to escape local minima
        if step % 10 == 0 and step > 0:
            noise = np.random.randn(len(params)) * 0.001
            noise[0] = 0.0
            params = params + noise
            params[0] = max(params[0], 0.01)
            
    r_opt = params[0]
    c_opt = params[1:].reshape(n, 2)
    
    # 5. Safety adjustment to guarantee validity
    x = c_opt[:, 0]
    y = c_opt[:, 1]
    min_gap = np.min([
        np.min(x - r_opt), np.min(1.0 - x - r_opt),
        np.min(y - r_opt), np.min(1.0 - y - r_opt)
    ])
    
    diff = c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 2.0)
    min_gap = min(min_gap, np.min(dists - 2.0 * r_opt))
    
    if min_gap < 0:
        r_opt += min_gap / 2.0 - 1e-7
        
    radii = np.full(n, max(r_opt, 0.0))
    return c_opt, radii, float(np.sum(radii))
