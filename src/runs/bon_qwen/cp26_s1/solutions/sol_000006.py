# sol_000006 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6882cd8b) state=5bd7c2a9 sum of radii=2.625166 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def penalty_objective(vars, n, mu):
    cx = vars[:n]
    cy = vars[n:2*n]
    r = vars[2*n:]
    
    obj = -np.sum(r)
    
    # Boundary penalties
    p_b = np.sum(np.maximum(0, r - cx)**2 + 
                 np.maximum(0, r - (1 - cx))**2 +
                 np.maximum(0, r - cy)**2 + 
                 np.maximum(0, r - (1 - cy))**2)
                 
    # Overlap penalties
    diff_x = cx[:, None] - cx[None, :]
    diff_y = cy[:, None] - cy[None, :]
    dist = np.sqrt(diff_x**2 + diff_y**2)
    np.fill_diagonal(dist, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    p_o = np.sum(np.maximum(0, r_sum - dist)**2)
    
    return obj + mu * (p_b + p_o)

def compute_penalty(vars, n):
    cx = vars[:n]
    cy = vars[n:2*n]
    r = vars[2*n:]
    
    p_b = np.sum(np.maximum(0, r - cx)**2 + 
                 np.maximum(0, r - (1 - cx))**2 +
                 np.maximum(0, r - cy)**2 + 
                 np.maximum(0, r - (1 - cy))**2)
                 
    diff_x = cx[:, None] - cx[None, :]
    diff_y = cy[:, None] - cy[None, :]
    dist = np.sqrt(diff_x**2 + diff_y**2)
    np.fill_diagonal(dist, np.inf)
    
    r_sum = r[:, None] + r[None, :]
    p_o = np.sum(np.maximum(0, r_sum - dist)**2)
    
    return p_b + p_o

def project_to_feasible(vars, n):
    cx = vars[:n].copy()
    cy = vars[n:2*n].copy()
    r = vars[2*n:].copy()
    
    # Clamp to boundaries
    r = np.minimum(r, cx)
    r = np.minimum(r, 1 - cx)
    r = np.minimum(r, cy)
    r = np.minimum(r, 1 - cy)
    
    # Clamp to non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
            r_sum = r[i] + r[j]
            if r_sum > d:
                excess = r_sum - d
                r[i] -= excess / 2.0
                r[j] -= excess / 2.0
                
    r = np.maximum(r, 0.0)
    return np.concatenate([cx, cy, r])

def run_packing():
    np.random.seed(42)
    n = 26
    best_vars = None
    best_sum_r = -np.inf
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # Generate initial configurations
    inits = []
    # 1. 5x5 grid + center
    x_grid = np.linspace(0.1, 0.9, 5)
    y_grid = np.linspace(0.1, 0.9, 5)
    cx_g = np.tile(x_grid, 5)
    cy_g = np.repeat(y_grid, 5)
    cx_g = np.append(cx_g, 0.5)
    cy_g = np.append(cy_g, 0.5)
    inits.append((cx_g, cy_g))
    
    # 2. Random perturbations
    for _ in range(8):
        cx_r = np.random.uniform(0.15, 0.85, n)
        cy_r = np.random.uniform(0.15, 0.85, n)
        inits.append((cx_r, cy_r))
        
    for cx0, cy0 in inits:
        r0 = np.full(n, 0.09)
        vars0 = np.concatenate([cx0, cy0, r0])
        
        curr = vars0
        mu = 50.0
        
        # Iterative penalty increase to guide optimizer
        for _ in range(4):
            res = minimize(penalty_objective, curr, args=(n, mu), method='L-BFGS-B', bounds=bounds,
                           options={'ftol': 1e-10, 'gtol': 1e-8, 'maxiter': 5000})
            curr = res.x
            mu *= 10.0
            
        pen = compute_penalty(curr, n)
        if pen < 1e-4:
            valid_vars = project_to_feasible(curr, n)
            curr_sum = np.sum(valid_vars[2*n:])
            if curr_sum > best_sum_r:
                best_sum_r = curr_sum
                best_vars = valid_vars
                
    if best_vars is None:
        # Fallback to small valid packing
        cx_f = np.tile(np.linspace(0.1, 0.9, 5), 5)
        cy_f = np.repeat(np.linspace(0.1, 0.9, 5), 5)
        cx_f = np.append(cx_f, 0.5)
        cy_f = np.append(cy_f, 0.5)
        r_f = np.full(n, 0.05)
        best_vars = np.concatenate([cx_f, cy_f, r_f])
        
    cx = best_vars[:n]
    cy = best_vars[n:2*n]
    r = best_vars[2*n:]
    
    centers = np.column_stack([cx, cy])
    return centers, r, np.sum(r)
