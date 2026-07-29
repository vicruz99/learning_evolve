# sol_000006 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 77dfa116) state=77e65c22 sum of radii=1.954515 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def boundary_constraints(vars):
    """Enforces circles to stay inside [0,1]x[0,1]"""
    n = 26
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    return np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])

def overlap_constraints(vars):
    """Enforces non-overlapping circles using vectorized pairwise distances"""
    n = 26
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    triu_idx = np.triu_indices(n, k=1)
    return dist[triu_idx] - r[triu_idx[0]] - r[triu_idx[1]]

def objective_func(vars):
    """Negative sum of radii (to be minimized)"""
    return -np.sum(vars[2::3])

def run_packing():
    n = 26
    
    # Initial positions: hexagonal grid
    centers = []
    r_init = 0.07
    dy = r_init * 1.73205081
    dx = 2 * r_init
    row = 0
    while len(centers) < n:
        y = r_init + row * dy
        if y + r_init > 1.0:
            break
        col = 0
        while len(centers) < n:
            x = r_init + col * dx + (row % 2) * (dx / 2)
            if x + r_init > 1.0:
                break
            centers.append([x, y])
            col += 1
        row += 1
    centers = np.array(centers[:n])
    
    vars0 = np.zeros(3 * n)
    vars0[0::3] = centers[:, 0]
    vars0[1::3] = centers[:, 1]
    vars0[2::3] = r_init
    
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    cons = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': overlap_constraints}
    ]
    
    best_result = None
    best_obj = np.inf
    
    # Multiple restarts to escape local minima
    for seed in range(5):
        np.random.seed(seed)
        pert = np.random.uniform(-0.02, 0.02, size=vars0.shape)
        vars_pert = vars0 + pert
        vars_pert[:2*n] = np.clip(vars_pert[:2*n], 0.01, 0.99)
        vars_pert[2*n:] = np.clip(vars_pert[2*n:], 0.01, 0.4)
        
        result = minimize(objective_func, vars_pert, method='SLSQP', bounds=bounds, constraints=cons, 
                          options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
        
        if result.success and result.fun < best_obj:
            best_obj = result.fun
            best_result = result
            
    # Fallback if all perturbations failed
    if best_result is None:
        best_result = minimize(objective_func, vars0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 5000, 'ftol': 1e-12})
        
    vars_opt = best_result.x
    centers_opt = np.column_stack((vars_opt[0::3], vars_opt[1::3]))
    radii_opt = vars_opt[2::3]
    
    # Ensure non-negative radii (handles potential numerical noise)
    radii_opt = np.maximum(radii_opt, 0)
    
    return centers_opt, radii_opt, np.sum(radii_opt)
