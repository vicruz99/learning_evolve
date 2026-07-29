# sol_000048 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1f1389a1) state=d5ab7880 sum of radii=2.598563 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v, n):
    # Maximize sum of radii => minimize negative sum
    # Radii are stored at indices 2, 5, 8, ... in the flat vector v
    return -np.sum(v[2::3])

def bound_constraint_x_min(v, i):
    return v[3*i] - v[3*i+2]

def bound_constraint_x_max(v, i):
    return 1.0 - v[3*i] - v[3*i+2]

def bound_constraint_y_min(v, i):
    return v[3*i+1] - v[3*i+2]

def bound_constraint_y_max(v, i):
    return 1.0 - v[3*i+1] - v[3*i+2]

def overlap_constraint(v, i, j):
    dx = v[3*i] - v[3*j]
    dy = v[3*i+1] - v[3*j+1]
    r_sum = v[3*i+2] + v[3*j+2]
    return dx*dx + dy*dy - r_sum*r_sum

def run_packing():
    n = 26
    
    # Hexagonal initialization for dense, valid starting configuration
    r_init = 0.04
    centers = []
    row_counts = [6, 5, 6, 5, 4]
    y = r_init
    for idx, count in enumerate(row_counts):
        step = 2 * r_init
        if count == 6:
            xs = np.linspace(r_init, 1 - r_init, 6)
        elif count == 5:
            xs = np.linspace(r_init + step, 1 - r_init - step, 5)
        else: 
            xs = np.linspace(r_init + step/2, 1 - r_init - step/2, 4)
        for x in xs:
            centers.append([x, y])
        y += np.sqrt(3) * r_init
        
    centers = np.array(centers)
    radii = np.full(n, r_init)
    
    # Flatten to [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': bound_constraint_x_min, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_constraint_x_max, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_constraint_y_min, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': bound_constraint_y_max, 'args': (i,)})
        
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': overlap_constraint, 'args': (i, j)})
            
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 1.0)] * n
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                   constraints=constraints, args=(n,),
                   options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
                   
    best_v = res.x if res.success else x0
        
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    for i in range(n):
        best_centers[i, 0] = best_v[3*i]
        best_centers[i, 1] = best_v[3*i+1]
        best_radii[i] = max(0.0, best_v[3*i+2])
        
    # Apply a tiny safety margin to strictly satisfy validator tolerances against numerical drift
    best_radii *= (1.0 - 1e-7)
    
    return best_centers, best_radii, np.sum(best_radii)
