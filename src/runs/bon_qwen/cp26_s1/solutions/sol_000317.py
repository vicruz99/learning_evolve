# sol_000317 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 78c934c9) state=9b45f35a sum of radii=2.549783 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import functools

def obj_func(v, n):
    """Objective: minimize negative sum of radii"""
    return -np.sum(v[2*n:])

def bound_x_min(v, i, n): return v[2*i] - v[2*n+i]
def bound_y_min(v, i, n): return v[2*i+1] - v[2*n+i]
def bound_x_max(v, i, n): return 1.0 - v[2*i] - v[2*n+i]
def bound_y_max(v, i, n): return 1.0 - v[2*i+1] - v[2*n+i]

def non_overlap(v, i, j, n):
    """Constraint: squared distance between centers >= squared sum of radii"""
    dx = v[2*i] - v[2*j]
    dy = v[2*i+1] - v[2*j+1]
    dr = v[2*n+i] + v[2*n+j]
    return dx*dx + dy*dy - dr*dr

def run_packing():
    n = 26
    # Initial grid configuration: 6x5 grid, take first 26 positions
    cols, rows = 6, 5
    centers = np.zeros((n, 2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                centers[idx] = [(c + 0.5) / cols, (r + 0.5) / rows]
                idx += 1
                
    # Feasible starting radii
    radii = np.full(n, 0.08)
    x0 = np.hstack([centers.flatten(), radii])
    
    # Variable bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    cons = []
    # Add boundary constraints
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': functools.partial(bound_x_min, i=i, n=n)})
        cons.append({'type': 'ineq', 'fun': functools.partial(bound_y_min, i=i, n=n)})
        cons.append({'type': 'ineq', 'fun': functools.partial(bound_x_max, i=i, n=n)})
        cons.append({'type': 'ineq', 'fun': functools.partial(bound_y_max, i=i, n=n)})
        
    # Add pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': functools.partial(non_overlap, i=i, j=j, n=n)})
            
    # Optimize
    res = minimize(obj_func, x0, args=(n,), method='SLSQP', bounds=bounds,
                   constraints=cons, options={'maxiter': 3000, 'ftol': 1e-10})
                   
    v_opt = res.x
    centers_opt = v_opt[:2*n].reshape((n, 2))
    radii_opt = v_opt[2*n:]
    
    # Ensure strict feasibility against numerical noise
    radii_opt = np.maximum(radii_opt, 0.0)
    centers_opt = np.clip(centers_opt, 0, 1)
    
    return centers_opt, radii_opt, np.sum(radii_opt)
