# sol_000020 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfb3fe63) state=5fb87e12 sum of radii=1.040000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_func(x, n):
    """Objective: minimize negative sum of radii"""
    return -np.sum(x[:n])

def boundary_func(x, n):
    """Constraints: circles inside [0,1]x[0,1]"""
    r = x[:n]
    cx = x[n:2*n]
    cy = x[2*n:3*n]
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    return np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])

def pairwise_func(x, n):
    """Constraints: non-overlapping circles"""
    r = x[:n]
    cx = x[n:2*n]
    cy = x[2*n:3*n]
    
    # Broadcast to n x n matrices
    r_exp = r[:, np.newaxis]
    cx_exp = cx[:, np.newaxis]
    cy_exp = cy[:, np.newaxis]
    
    # Sum of radii and squared distances
    dr = r_exp + r_exp.T
    dxc = cx_exp - cx_exp.T
    dyc = cy_exp - cy_exp.T
    
    d2 = dxc**2 + dyc**2
    rsq = dr**2
    
    # Extract upper triangle (i < j)
    idx = np.triu_indices(n, k=1)
    # Constraint: dist^2 - (r_i + r_j)^2 >= 0
    return d2[idx] - rsq[idx]

def run_packing():
    n = 26
    best_x = None
    best_val = -np.inf

    # Base hexagonal grid initialization
    ys = np.linspace(0.15, 0.85, 6)
    xs_e = np.linspace(0.15, 0.85, 5)
    xs_o = np.linspace(0.225, 0.775, 4)
    pos_list = []
    for i, y in enumerate(ys):
        xs = xs_e if i % 2 == 0 else xs_o
        for x in xs:
            pos_list.append([x, y])
        if len(pos_list) >= n:
            break
    base_pos = np.array(pos_list[:n])

    # Variable bounds: r > 0, x,y in [0,1]
    bounds = [(1e-6, 0.5)] * n + [(0.0, 1.0)] * (2 * n)
    
    # Constraint definitions
    cons = [
        {'type': 'ineq', 'fun': boundary_func, 'args': (n,)},
        {'type': 'ineq', 'fun': pairwise_func, 'args': (n,)}
    ]

    # Multi-restart optimization
    for seed in range(20):
        rng = np.random.RandomState(seed)
        
        # Perturb initial positions and radii
        pos = base_pos.copy()
        pos += rng.uniform(-0.03, 0.03, pos.shape)
        pos = np.clip(pos, 0.05, 0.95)
        
        r = 0.04 * np.ones(n) + rng.uniform(-0.005, 0.005, n)
        r = np.clip(r, 0.02, 0.08)

        x0 = np.concatenate([r, pos[:, 0], pos[:, 1]])
        
        try:
            res = minimize(obj_func, x0, args=(n,), method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 2500, 'ftol': 1e-11, 'disp': False})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    # Fallback if optimization fails completely
    if best_x is None:
        r = 0.04 * np.ones(n)
        best_x = np.concatenate([r, base_pos[:, 0], base_pos[:, 1]])

    r_final = best_x[:n]
    cx_final = best_x[n:2*n]
    cy_final = best_x[2*n:3*n]
    centers = np.column_stack([cx_final, cy_final])
    total_sum = np.sum(r_final)

    return centers, r_final, total_sum
