# sol_000301 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d28721c0) state=f75b542a sum of radii=2.440106 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_objective(vars):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    return -np.sum(vars[:N_CIRCLES])

def compute_constraints(vars):
    """Inequality constraints: g(x) >= 0"""
    n = N_CIRCLES
    r = vars[:n]
    c = vars[n:].reshape((n, 2))
    cons = []
    
    # Boundary constraints: center must be at least r away from edges
    for i in range(n):
        cons.append(c[i, 0] - r[i])          # x >= r
        cons.append(1.0 - r[i] - c[i, 0])    # x <= 1 - r
        cons.append(c[i, 1] - r[i])          # y >= r
        cons.append(1.0 - r[i] - c[i, 1])    # y <= 1 - r
        
    # Non-overlap constraints: distance^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i+1, n):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            dist2 = dx*dx + dy*dy
            sum_r = r[i] + r[j]
            cons.append(dist2 - sum_r**2)
            
    return np.array(cons)

def run_packing():
    n = N_CIRCLES
    r_init = 0.04
    centers_init = np.zeros((n, 2))
    radii_init = np.full(n, r_init)
    
    # Hexagonal lattice initialization
    idx = 0
    row = 0
    while idx < n:
        y = r_init + row * r_init * np.sqrt(3)
        if y + r_init > 0.99:
            break
        # Offset odd rows by one circle radius horizontally
        x_start = r_init + (0.5 if row % 2 == 1 else 0) * 2 * r_init
        for i in range(10):
            if idx >= n: break
            x = x_start + i * 2 * r_init
            if x + r_init <= 0.99:
                centers_init[idx] = [x, y]
                idx += 1
        row += 1
        
    # Combine radii and centers into a single optimization vector
    x0 = np.concatenate([radii_init, centers_init.flatten()])
    
    # Bounds: radii in [0.01, 0.5], centers in [0, 1]
    bounds = []
    for _ in range(n):
        bounds.append((0.01, 0.5))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        
    # Solve constrained optimization problem
    res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds, 
                   constraints={'type': 'ineq', 'fun': compute_constraints}, 
                   options={'maxiter': 3000, 'ftol': 1e-12})
                   
    if res.success:
        r_opt = res.x[:n]
        c_opt = res.x[n:].reshape((n, 2))
    else:
        # Fallback to initial configuration if optimization fails
        r_opt = radii_init
        c_opt = centers_init
        
    return c_opt, r_opt, float(np.sum(r_opt))
