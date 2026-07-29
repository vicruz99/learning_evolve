# sol_000038 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2740085f) state=a4bf7a4a sum of radii=2.614243 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(params):
    # Maximize sum of radii => minimize negative sum
    return -np.sum(params[2::3])

def boundary_con_func(params):
    n = len(params) // 3
    vals = np.empty(4 * n)
    for i in range(n):
        idx = 3 * i
        x, y, r = params[idx], params[idx+1], params[idx+2]
        # Inequality constraints: fun >= 0
        vals[4*i]   = x - r
        vals[4*i+1] = 1.0 - x - r
        vals[4*i+2] = y - r
        vals[4*i+3] = 1.0 - y - r
    return vals

def overlap_con_func(params):
    n = len(params) // 3
    m = n * (n - 1) // 2
    vals = np.empty(m)
    k = 0
    for i in range(n):
        xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
        for j in range(i + 1, n):
            xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
            dx, dy = xi - xj, yi - yj
            # dist^2 >= (ri + rj)^2
            vals[k] = dx*dx + dy*dy - (ri + rj)**2
            k += 1
    return vals

def run_packing():
    n = 26
    np.random.seed(42)
    
    # Initial guess: perturbed 5x5 grid + 1 center circle
    init_pts = []
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    for y in ys:
        for x in xs:
            init_pts.append([x, y])
    init_pts.append([0.5, 0.5])
    
    # Add small deterministic noise to break symmetry and aid optimization
    for i in range(n):
        init_pts[i][0] += np.random.normal(0, 0.005)
        init_pts[i][1] += np.random.normal(0, 0.005)
        
    r_start = 0.07
    params0 = np.zeros(3 * n)
    for i in range(n):
        params0[3*i] = init_pts[i][0]
        params0[3*i+1] = init_pts[i][1]
        params0[3*i+2] = r_start
        
    # Variable bounds
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    cons = [
        {'type': 'ineq', 'fun': boundary_con_func},
        {'type': 'ineq', 'fun': overlap_con_func}
    ]
    
    # Optimize
    res = minimize(objective, params0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
    
    params_opt = res.x
    
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i] = params_opt[3*i:3*i+2]
        radii[i] = params_opt[3*i+2]
        
    # Apply tiny safety margin to guarantee strict feasibility
    radii = np.maximum(0.0, radii - 1e-7)
    
    sum_r = float(np.sum(radii))
    return centers, radii, sum_r
