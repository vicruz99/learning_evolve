# sol_000107 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 81b841bb) state=93d40359 sum of radii=2.453842 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n):
    """Objective function for circle packing optimization."""
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Primary objective: maximize sum of radii
    obj = -np.sum(r)
    pen = 0.0
    
    # Boundary constraints: circle i must be inside [0,1]x[0,1]
    pen += 800.0 * np.sum(np.maximum(0.0, r - c[:, 0])**2)
    pen += 800.0 * np.sum(np.maximum(0.0, r - c[:, 1])**2)
    pen += 800.0 * np.sum(np.maximum(0.0, c[:, 0] + r - 1.0)**2)
    pen += 800.0 * np.sum(np.maximum(0.0, c[:, 1] + r - 1.0)**2)
    
    # Pairwise non-overlap constraints
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # High weight for overlaps to ensure strict separation
    pen += 2000.0 * np.sum(np.maximum(0.0, r[i_idx] + r[j_idx] - dists[i_idx, j_idx])**2)
    
    return obj + pen

def run_packing():
    n = 26
    
    # 1. Hexagonal lattice initialization
    # Rows: 6, 5, 6, 5, 4 circles total 26
    rows = [6, 5, 6, 5, 4]
    r_init = 0.09
    w = 2.0 * r_init
    h = w * np.sqrt(3.0) / 2.0
    
    centers = []
    for i, count in enumerate(rows):
        y = i * h + r_init
        x_start = r_init + (i % 2) * w / 2.0
        for j in range(count):
            x = x_start + j * w
            centers.append([x, y])
            
    centers = np.array(centers)
    radii = np.full(n, r_init)
    
    # Flatten for optimizer
    x0 = np.concatenate([centers.ravel(), radii])
    
    # Box constraints: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # 2. Gradient-based optimization
    res = minimize(
        compute_objective,
        x0,
        args=(n,),
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 50000,
            'ftol': 1e-15,
            'gtol': 1e-12
        }
    )
    
    best_c = res.x[:2*n].reshape(n, 2)
    best_r = res.x[2*n:]
    
    # 3. Constraint tightening & validation guarantee
    best_c = np.clip(best_c, 1e-9, 1.0 - 1e-9)
    best_r = np.clip(best_r, 1e-9, 0.5)
    
    # Enforce boundary constraints exactly
    for i in range(n):
        limit = min(best_c[i, 0], 1.0 - best_c[i, 0], 
                    best_c[i, 1], 1.0 - best_c[i, 1])
        best_r[i] = min(best_r[i], limit - 1e-9)
        
    # Iteratively resolve any remaining overlaps
    for _ in range(200):
        overlap = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((best_c[i] - best_c[j])**2))
                if d < best_r[i] + best_r[j] + 1e-9:
                    overlap = True
                    shrink = (best_r[i] + best_r[j] - d + 2e-9) / 2.0
                    best_r[i] -= shrink
                    best_r[j] -= shrink
        if not overlap:
            break
            
    # Ensure non-negative radii after shrinking
    best_r = np.maximum(best_r, 1e-9)
    
    return best_c, best_r, float(np.sum(best_r))
