# sol_000020 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a75b8609) state=6f2d6856 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    n = 26
    radii = vars[2::3]
    return -np.sum(radii)

def get_constraints(n):
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        # x_i >= r_i
        def c_x_ge_r(vars, idx=i):
            return vars[3*idx] - vars[3*idx + 2]
        constraints.append({'type': 'ineq', 'fun': c_x_ge_r})
        
        # x_i + r_i <= 1
        def c_x_plus_r_le_1(vars, idx=i):
            return 1 - vars[3*idx] - vars[3*idx + 2]
        constraints.append({'type': 'ineq', 'fun': c_x_plus_r_le_1})
        
        # y_i >= r_i
        def c_y_ge_r(vars, idx=i):
            return vars[3*idx + 1] - vars[3*idx + 2]
        constraints.append({'type': 'ineq', 'fun': c_y_ge_r})
        
        # y_i + r_i <= 1
        def c_y_plus_r_le_1(vars, idx=i):
            return 1 - vars[3*idx + 1] - vars[3*idx + 2]
        constraints.append({'type': 'ineq', 'fun': c_y_plus_r_le_1})

    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def c_no_overlap(vars, idx_i=i, idx_j=j):
                xi = vars[3*idx_i]
                yi = vars[3*idx_i + 1]
                ri = vars[3*idx_i + 2]
                
                xj = vars[3*idx_j]
                yj = vars[3*idx_j + 1]
                rj = vars[3*idx_j + 2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r_sq = (ri + rj)**2
                
                return dist_sq - sum_r_sq
            
            constraints.append({'type': 'ineq', 'fun': c_no_overlap})
            
    return constraints

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    r_init = 0.09
    centers_list = []
    
    dy = np.sqrt(3) * r_init
    dx = 2 * r_init
    
    row = 0
    y = r_init
    
    while y + r_init <= 1.0 + 1e-9:
        if row % 2 == 0:
            x = r_init
            while x + r_init <= 1.0 + 1e-9:
                centers_list.append([x, y])
                x += dx
        else:
            x = 2 * r_init
            while x + r_init <= 1.0 + 1e-9:
                centers_list.append([x, y])
                x += dx
        
        y += dy
        row += 1
        
        if len(centers_list) >= n:
            break
            
    if len(centers_list) > n:
        centers_list = centers_list[:n]
    while len(centers_list) < n:
        centers_list.append([0.5, 0.5])
        
    centers_init = np.array(centers_list[:n])
    radii_init = np.full(n, r_init)
    
    # Flatten variables
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i + 1] = centers_init[i, 1]
        x0[3*i + 2] = radii_init[i]
        
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
        
    constraints = get_constraints(n)
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'ftol': 1e-9, 'maxiter': 2000})
        x_opt = res.x
    except Exception:
        x_opt = x0
        
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = x_opt[3*i]
        centers[i, 1] = x_opt[3*i + 1]
        radii[i] = x_opt[3*i + 2]
        
    radii = np.maximum(radii, 0)
    
    # Post-processing to ensure validity
    for i in range(n):
        r = radii[i]
        x, y = centers[i]
        centers[i, 0] = np.clip(x, r, 1 - r)
        centers[i, 1] = np.clip(y, r, 1 - r)
        
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
