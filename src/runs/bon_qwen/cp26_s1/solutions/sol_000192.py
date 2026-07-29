# sol_000192 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 320c78c6) state=1dc7a637 sum of radii=0.000325 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
LAMBDA = 10000.0

def compute_loss(vars, n, lam):
    x = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    sum_r = np.sum(r)
    
    # Overlap penalty
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    overlap = r_sum - dists
    penalty_overlap = np.sum(np.maximum(0, overlap)**2)
    
    # Boundary penalty
    x_coords = x[:, 0]
    y_coords = x[:, 1]
    
    viol_left = np.maximum(0, r - x_coords)
    viol_right = np.maximum(0, x_coords + r - 1.0)
    viol_bottom = np.maximum(0, r - y_coords)
    viol_top = np.maximum(0, y_coords + r - 1.0)
    
    penalty_boundary = np.sum(viol_left**2) + np.sum(viol_right**2) + np.sum(viol_bottom**2) + np.sum(viol_top**2)
    
    total_penalty = penalty_overlap + penalty_boundary
    
    return -sum_r + lam * total_penalty

def get_initial_config(n):
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx, 0] = 0.1 + 0.2 * i
            centers[idx, 1] = 0.1 + 0.2 * j
            radii[idx] = 0.1
            idx += 1
            
    centers[25, 0] = 0.2
    centers[25, 1] = 0.2
    radii[25] = 0.04
    
    return centers, radii

def fix_validity(centers, radii, n):
    f = 1.0
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r <= 1e-9: continue
        val = x / r
        if val < f: f = val
        val = (1.0 - x) / r
        if val < f: f = val
        val = y / r
        if val < f: f = val
        val = (1.0 - y) / r
        if val < f: f = val
        
    for i in range(n):
        for j in range(i + 1, n):
            r_sum = radii[i] + radii[j]
            if r_sum <= 1e-9: continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            val = dist / r_sum
            if val < f: f = val
            
    if f < 0: f = 0.0
    if f < 1.0 - 1e-9:
        radii = radii * f
        
    return radii

def run_packing():
    n = 26
    centers, radii = get_initial_config(n)
    x0 = np.concatenate([centers.flatten(), radii])
    
    bounds = []
    for _ in range(2 * n):
        bounds.append((0.0, 1.0))
    for _ in range(n):
        bounds.append((0.0, 0.5))
        
    res = minimize(compute_loss, x0, args=(n, LAMBDA), method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-6})
    
    final_vars = res.x
    if np.isnan(final_vars).any():
        return centers, radii, np.sum(radii)
        
    final_centers = final_vars[:2*n].reshape(n, 2)
    final_radii = final_vars[2*n:]
    
    final_radii = fix_validity(final_centers, final_radii, n)
    
    total_sum = np.sum(final_radii)
    
    return final_centers, final_radii, total_sum
