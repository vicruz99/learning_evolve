# sol_000168 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1d84d4eb) state=5971a125 sum of radii=0.026000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_centers_vectorized(u, v, r):
    """Compute centers from transformed variables u, v, r."""
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return np.column_stack((x, y))

def loss_function(vars, n, penalty):
    """Objective function: maximize sum of radii with overlap penalty."""
    u = vars[:n]
    v = vars[n:2*n]
    r = vars[2*n:]
    
    c = compute_centers_vectorized(u, v, r)
    obj = -np.sum(r)
    
    # Vectorized pairwise distances
    c_diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(c_diff**2, axis=2))
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Penalty for overlaps
    violation = np.maximum(0.0, r_sum - dists)
    obj += penalty * np.sum(violation**2)
    
    return obj

def run_packing():
    n = 26
    
    # 1. Initial configuration: Hexagonal-like arrangement
    centers_init = np.zeros((n, 2))
    r_init = np.full(n, 0.08)
    idx = 0
    row_counts = [5, 5, 5, 5, 6]  # Total 26 circles
    y_step = 1.0 / 6
    for i, count in enumerate(row_counts):
        y = (i + 1) * y_step
        x_step = 1.0 / (count + 1)
        shift = x_step * 0.5 if i % 2 == 1 else 0.0
        for j in range(count):
            x = (j + 1) * x_step + shift
            centers_init[idx] = [x, y]
            idx += 1
            
    # 2. Transform to u, v space for boundary-safe optimization
    init_u = np.zeros(n)
    init_v = np.zeros(n)
    for i in range(n):
        cx, cy = centers_init[i]
        r = r_init[i]
        denom = 1.0 - 2.0 * r
        if denom > 1e-6:
            init_u[i] = np.clip((cx - r) / denom, 0.0, 1.0)
            init_v[i] = np.clip((cy - r) / denom, 0.0, 1.0)
        else:
            init_u[i] = 0.5
            init_v[i] = 0.5
            
    x0 = np.concatenate([init_u, init_v, r_init])
    bounds = [(0.0, 1.0)] * (2*n) + [(0.001, 0.4)] * n
    
    # 3. Iterative Penalty Optimization
    current_x = x0
    penalty = 500.0
    for k in range(12):
        res = minimize(loss_function, current_x, args=(n, penalty), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 600, 'ftol': 1e-12, 'gtol': 1e-10})
        current_x = res.x
        penalty *= 1.8
        
    # 4. Extract and finalize
    u = current_x[:n]
    v = current_x[n:2*n]
    r = current_x[2*n:]
    
    centers = compute_centers_vectorized(u, v, r)
    
    # Safety clamping to satisfy validation strictly
    r = np.maximum(r, 1e-9)
    centers[:, 0] = np.clip(centers[:, 0], r, 1.0 - r)
    centers[:, 1] = np.clip(centers[:, 1], r, 1.0 - r)
    
    sum_r = np.sum(r)
    return centers, r, sum_r
