# sol_000312 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1cbfbe8a) state=4843e220 sum of radii=2.491296 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars, n):
    """
    Objective function for circle packing optimization.
    Minimizes -sum(radii) + penalty for constraint violations.
    """
    cx = vars[:2*n]
    r = vars[2*n:]
    
    # We want to maximize sum(r), so minimize -sum(r)
    val = -np.sum(r)
    pen = 0.0
    
    # Boundary penalties
    for i in range(n):
        xi, yi = cx[2*i], cx[2*i+1]
        ri = r[i]
        # Left
        if xi < ri:
            pen += 1000.0 * (ri - xi)**2
        # Right
        if xi + ri > 1.0:
            pen += 1000.0 * (xi + ri - 1.0)**2
        # Bottom
        if yi < ri:
            pen += 1000.0 * (ri - yi)**2
        # Top
        if yi + ri > 1.0:
            pen += 1000.0 * (yi + ri - 1.0)**2
            
    # Overlap penalties
    C = cx.reshape(n, 2)
    # Pairwise differences
    diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1.0) # Ignore self-distance
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    violations = r_sum - dists
    np.fill_diagonal(violations, -1.0)
    
    # Only consider upper triangle to avoid double counting
    triu_idx = np.triu_indices(n, k=1)
    viol_tri = violations[triu_idx]
    pen += 1000.0 * np.sum(np.square(np.maximum(0.0, viol_tri)))
    
    return val + pen

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    r_init = 0.085
    
    # Initial hexagonal configuration
    # Row counts chosen to compactly fit 26 circles
    rows = [6, 5, 6, 5, 4]
    centers = []
    h = r_init * np.sqrt(3)
    for i, count in enumerate(rows):
        y = r_init + i * h
        width_needed = 2 * count * r_init
        shift = (1.0 - width_needed) / 2.0
        for j in range(count):
            x = shift + r_init + j * 2 * r_init
            centers.append([x, y])
    centers = np.array(centers)
    
    # Variables: x, y, r for each circle
    x0 = np.concatenate([centers.flatten(), np.full(n, r_init)])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # Optimization
    res = minimize(objective, x0, args=(n,), method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-8})
    
    best_vars = res.x
    best_centers = best_vars[:2*n].reshape(n, 2)
    best_radii = best_vars[2*n:]
    
    # Post-processing to guarantee strict validity
    # Iteratively enforce constraints to handle any numerical slack
    for _ in range(5):
        # Enforce boundaries
        for i in range(n):
            xi, yi = best_centers[i]
            ri = best_radii[i]
            best_radii[i] = min(ri, xi, 1.0 - xi, yi, 1.0 - yi)
            
        # Enforce non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                if best_radii[i] + best_radii[j] > d - 1e-12:
                    overlap = best_radii[i] + best_radii[j] - d + 1e-9
                    best_radii[i] -= overlap / 2.0
                    best_radii[j] -= overlap / 2.0
                    
    best_radii = np.maximum(best_radii, 0.0)
    
    return best_centers, best_radii, float(np.sum(best_radii))
