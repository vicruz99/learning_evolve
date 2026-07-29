# sol_000286 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0a16b6e7) state=b0b0d11a sum of radii=0.021667 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars_flat):
    """
    Computes the negative sum of radii plus penalties for constraint violations.
    """
    n = 26
    centers = vars_flat[:2*n].reshape(n, 2)
    radii = vars_flat[2*n:]
    
    # Primary objective: maximize sum of radii
    obj = -np.sum(radii)
    
    # Penalty weight
    w = 150.0
    
    # Boundary penalties: circles must stay within [0,1]x[0,1]
    pen_x_left = np.maximum(0, radii - centers[:, 0])**2
    pen_y_left = np.maximum(0, radii - centers[:, 1])**2
    pen_x_right = np.maximum(0, radii + centers[:, 0] - 1.0)**2
    pen_y_right = np.maximum(0, radii + centers[:, 1] - 1.0)**2
    boundary_pen = w * np.sum(pen_x_left + pen_y_left + pen_x_right + pen_y_right)
    
    # Overlap penalties: distance between centers must be >= sum of radii
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 0)
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap = np.maximum(0, r_sum - dists)
    overlap_pen = w * np.sum(overlap**2)
    
    return obj + boundary_pen + overlap_pen

def run_packing():
    n = 26
    # Bounds for variables: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    inits = []
    
    # Configuration 1: 5x5 Grid + 1
    c1 = []
    for i in range(5):
        for j in range(5):
            c1.append([(i + 0.5) / 5.0, (j + 0.5) / 5.0])
    c1.append([0.5, 0.5])
    inits.append(np.hstack([np.array(c1).flatten(), np.ones(n) * 0.09]))
    
    # Configuration 2: Hexagonal-like arrangement
    c2 = []
    for r_idx in range(5):
        cnt = 6 if r_idx % 2 == 0 else 5
        y = (r_idx + 0.5) * 0.2
        for c_idx in range(cnt):
            x = (c_idx + 0.5) * (2.0 / max(cnt, 5)) + (0.05 if r_idx % 2 == 1 else 0)
            if len(c2) < 26:
                c2.append([x, y])
    while len(c2) < 26:
        c2.append([0.5, 0.5])
    inits.append(np.hstack([np.array(c2[:26]).flatten(), np.ones(n) * 0.09]))
    
    # Configuration 3: Random placement
    np.random.seed(42)
    c3 = np.random.uniform(0.1, 0.9, (n, 2))
    inits.append(np.hstack([c3.flatten(), np.ones(n) * 0.08]))
    
    best_res = None
    best_val = np.inf
    
    # Optimize from each initialization
    for init in inits:
        # Perturb to break symmetry
        init = init + np.random.uniform(-0.005, 0.005, init.shape)
        res = minimize(compute_objective, init, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 5000, 'ftol': 1e-12})
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
            
    if best_res is None:
        # Fallback
        centers = np.array(c1)
        radii = np.ones(n) * 0.09
        return centers, radii, float(np.sum(radii))
        
    centers = best_res.x[:2*n].reshape(n, 2)
    radii = best_res.x[2*n:]
    
    # Post-processing: enforce constraints strictly
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
    
    # Check for residual overlaps and scale down if necessary
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_ratio = np.min(np.where(r_sum > 1e-12, dists / r_sum, 1.0))
    
    if min_ratio < 1.0 - 1e-9:
        radii *= min_ratio
        
    # Final clamp after potential scaling
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
    
    return centers, radii, float(np.sum(radii))
