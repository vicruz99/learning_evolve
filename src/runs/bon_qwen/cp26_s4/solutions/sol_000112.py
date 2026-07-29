# sol_000112 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 34ab26db) state=e3891534 sum of radii=2.592205 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def con_x_min(v, i, n):
    return v[2*i] - v[2*n+i]

def con_x_max(v, i, n):
    return 1.0 - v[2*i] - v[2*n+i]

def con_y_min(v, i, n):
    return v[2*i+1] - v[2*n+i]

def con_y_max(v, i, n):
    return 1.0 - v[2*i+1] - v[2*n+i]

def con_dist(v, i, j, n):
    dx = v[2*i] - v[2*j]
    dy = v[2*i+1] - v[2*j+1]
    r_sum = v[2*n+i] + v[2*n+j]
    return np.sqrt(dx*dx + dy*dy) - r_sum

def objective(v, n):
    return -np.sum(v[2*n:])

def run_packing():
    n = 26
    
    # Hexagonal-like grid initialization for better convergence
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.07)
    idx = 0
    
    for row in range(5):
        y = 0.12 + row * 0.19
        for col in range(6):
            if idx >= n:
                break
            x = 0.08 + col * 0.16
            if row % 2 == 1:
                x += 0.08
            centers[idx] = [x, y]
            idx += 1
            
    while idx < n:
        centers[idx] = [0.5, 0.5]
        idx += 1
        
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0) for _ in range(2*n)] + [(0.0, 0.5) for _ in range(n)]
    
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': con_x_min, 'args': (i, n)})
        constraints.append({'type': 'ineq', 'fun': con_x_max, 'args': (i, n)})
        constraints.append({'type': 'ineq', 'fun': con_y_min, 'args': (i, n)})
        constraints.append({'type': 'ineq', 'fun': con_y_max, 'args': (i, n)})
        
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': con_dist, 'args': (i, j, n)})
            
    try:
        res = minimize(
            objective,
            x0,
            args=(n,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
        )
        x_opt = res.x
    except Exception:
        x_opt = x0
        
    centers_opt = x_opt[:2*n].reshape(n, 2)
    radii_opt = np.maximum(x_opt[2*n:], 0.0)
    
    return centers_opt, radii_opt, float(np.sum(radii_opt))
