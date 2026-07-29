# sol_000339 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2bd19375) state=3a443017 sum of radii=1.837376 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def calc_penalty(centers, radii):
    n = len(radii)
    # Vectorized inter-circle distance calculation
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Only check lower triangle to avoid double counting and self-distance
    i, j = np.tril_indices(n, -1)
    gaps = dists[i, j] - (radii[i] + radii[j])
    
    # Penalty for overlaps
    pen = np.sum(np.square(np.where(gaps < 0, gaps, 0)))
    
    # Boundary penalties
    x, y = centers[:, 0], centers[:, 1]
    r = radii
    pen += np.sum(np.square(np.where(x - r < 0, x - r, 0)))
    pen += np.sum(np.square(np.where(x + r - 1 > 0, x + r - 1, 0)))
    pen += np.sum(np.square(np.where(y - r < 0, y - r, 0)))
    pen += np.sum(np.square(np.where(y + r - 1 > 0, y + r - 1, 0)))
    
    return pen

def pos_objective(x, n, radii):
    c = x.reshape(n, 2)
    return calc_penalty(c, radii)

def joint_objective(v, n):
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    return -np.sum(r) + 2000.0 * calc_penalty(c, r)

def run_packing():
    n = 26
    best_sum = 0.0
    best_cfg = None
    np.random.seed(42)
    
    # Generate initial configurations
    configs = []
    
    # 1. Grid arrangement
    c = np.zeros((n, 2))
    gs = int(np.ceil(np.sqrt(n)))
    for idx in range(n):
        c[idx] = [(idx % gs) * (1.0/gs) + 0.5/gs, (idx // gs) * (1.0/gs) + 0.5/gs]
    configs.append(c)
    
    # 2. Hexagonal arrangement
    c = np.zeros((n, 2))
    idx = 0
    row = 0
    while idx < n:
        cols_in_row = gs if row % 2 == 0 else gs - 1
        for col in range(cols_in_row):
            if idx < n:
                c[idx] = [col / (gs - 0.5) + 0.5/(gs-0.5), row * 0.866 / (gs-1) + 0.5]
                idx += 1
        row += 1
    configs.append(c)
    
    # 3. Random arrangement
    c = np.random.rand(n, 2) * 0.8 + 0.1
    configs.append(c)
    
    # Normalize each config to be safely within [0.1, 0.9]
    init_configs = []
    for c in configs:
        c_min = c.min(axis=0)
        c_max = c.max(axis=0)
        c_norm = (c - c_min) / (c_max - c_min) * 0.8 + 0.1
        init_configs.append(c_norm)
        
    for init_centers in init_configs:
        radii = np.full(n, 0.015)
        current_centers = init_centers.copy()
        
        # Growing circles phase
        for step in range(120):
            radii *= 1.013
            
            res = minimize(pos_objective, current_centers.flatten(), method='L-BFGS-B',
                           args=(n, radii),
                           bounds=[(0, 1)] * (2*n), options={'maxiter': 400, 'ftol': 1e-10})
            current_centers = res.x.reshape(n, 2)
            
        s = np.sum(radii)
        if s > best_sum:
            best_sum = s
            best_cfg = (current_centers.copy(), radii.copy())
            
    # Joint optimization phase to maximize sum of radii
    if best_cfg is not None:
        centers, radii = best_cfg
        
        v0 = np.concatenate([centers.flatten(), radii])
        bounds = [(0, 1)] * (2*n) + [(1e-6, 0.5)] * n
        res = minimize(joint_objective, v0, method='L-BFGS-B', args=(n,), 
                       bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-12})
        
        final_centers = res.x[:2*n].reshape(n, 2)
        final_radii = res.x[2*n:]
        
        # Ensure strict validity with small tolerance
        for i in range(n):
            r = final_radii[i]
            final_centers[i, 0] = np.clip(final_centers[i, 0], r + 1e-8, 1 - r - 1e-8)
            final_centers[i, 1] = np.clip(final_centers[i, 1], r + 1e-8, 1 - r - 1e-8)
            
        best_sum = np.sum(final_radii)
        return final_centers, final_radii, best_sum
        
    return best_cfg[0], best_cfg[1], best_sum
