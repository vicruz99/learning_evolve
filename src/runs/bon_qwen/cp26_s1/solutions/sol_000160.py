# sol_000160 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5dc93b19) state=c6c2df88 sum of radii=2.457658 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars, n):
    """Objective function: minimize negative sum of radii"""
    r = vars[2::3]
    return -np.sum(r)

def constraint_dist(vars, n):
    """Non-overlap constraints: distance >= sum of radii"""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    diff_x = x[:, None] - x[None, :]
    diff_y = y[:, None] - y[None, :]
    dist = np.sqrt(diff_x**2 + diff_y**2)
    r_sum = r[:, None] + r[None, :]
    
    i, j = np.triu_indices(n, k=1)
    return dist[i, j] - r_sum[i, j]

def constraint_wall(vars, n):
    """Boundary constraints: circles inside [0,1]x[0,1]"""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    return np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])

def run_packing():
    n = 26
    
    # Feasible initialization: 5x5 grid + 1 center circle
    centers = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
            idx += 1
    centers[25] = [0.5, 0.5]
    
    # Flatten to optimization variable vector: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = 0.09  # Start slightly smaller than 0.1 to ensure feasibility
        
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-6, 0.5)] * n
    
    cons = [
        {'type': 'ineq', 'fun': constraint_dist, 'args': (n,)},
        {'type': 'ineq', 'fun': constraint_wall, 'args': (n,)}
    ]
    
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        args=(n,), 
        options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False}
    )
    
    centers_res = np.zeros((n, 2))
    radii_res = np.zeros(n)
    for i in range(n):
        centers_res[i] = [res.x[3*i], res.x[3*i+1]]
        radii_res[i] = res.x[3*i+2]
        
    return centers_res, radii_res, np.sum(radii_res)
