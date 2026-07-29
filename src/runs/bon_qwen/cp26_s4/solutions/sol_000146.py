# sol_000146 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 466799c7) state=438e610d sum of radii=2.344158 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_obj(x, n, P):
    """
    Computes the objective: negative sum of radii plus penalty for constraint violations.
    x: flattened array [cx1, cy1, r1, cx2, cy2, r2, ...]
    """
    cx = x[::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary violations: circle must be inside [0,1]x[0,1]
    viol = np.maximum(r - cx, 0.0)**2 + np.maximum(r - (1.0 - cx), 0.0)**2
    viol += np.maximum(r - cy, 0.0)**2 + np.maximum(r - (1.0 - cy), 0.0)**2
    
    # Pairwise violations: circles must not overlap
    pos = np.column_stack((cx, cy))
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    r_sum = r[:, None] + r[None, :]
    pair_viol = np.maximum(r_sum - dist, 0.0)**2
    np.fill_diagonal(pair_viol, 0.0)
    
    return -np.sum(r) + P * (np.sum(viol) + np.sum(pair_viol))

def run_packing():
    n = 26
    P = 10000.0
    
    # Initialization: Hexagonal-like grid pattern
    x0 = np.zeros(n * 3)
    idx = 0
    rows = [6, 5, 6, 5, 4]
    for row_i, n_cols in enumerate(rows):
        y = (row_i + 0.5) / 5.0
        # Shift every other row to create hexagonal packing
        shift = 0.5 / n_cols if row_i % 2 == 1 else 0.0
        for col_i in range(n_cols):
            x = (col_i + 0.5) / n_cols + shift
            x = np.clip(x, 0.0, 1.0)
            y = np.clip(y, 0.0, 1.0)
            x0[3 * idx] = x
            x0[3 * idx + 1] = y
            x0[3 * idx + 2] = 0.06  # Initial radius
            idx += 1
            
    # Bounds for L-BFGS-B
    bounds_flat = []
    for _ in range(n):
        bounds_flat.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    def objective(x):
        return compute_obj(x, n, P)
        
    # Optimize
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds_flat, 
                   options={'maxiter': 3000, 'ftol': 1e-12})
    
    cx = res.x[::3]
    cy = res.x[1::3]
    radii = res.x[2::3]
    
    # Feasibility correction loop
    # Ensures strict validity according to the validator's tolerance
    for _ in range(200):
        valid = True
        max_viol = 0.0
        
        # Check boundaries
        for i in range(n):
            for d in [cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i]]:
                if radii[i] > d + 1e-12:
                    valid = False
                    max_viol = max(max_viol, radii[i] - d)
                    
        # Check pairwise overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
                if radii[i] + radii[j] > dist + 1e-12:
                    valid = False
                    max_viol = max(max_viol, radii[i] + radii[j] - dist)
                    
        if valid:
            break
            
        # Uniformly shrink radii to resolve overlaps
        shrink = max_viol / 2.0 + 1e-8
        radii = np.maximum(radii - shrink, 0.0)
        
    centers = np.column_stack((cx, cy))
    return centers, radii, np.sum(radii)
