# sol_000237 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 06f8ea92) state=7d08ccc8 sum of radii=2.620159 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(vars):
    n = int(len(vars) / 3)
    return -np.sum(vars[2*n:])

def compute_constraints(vars):
    n = int(len(vars) / 3)
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    m = n*4 + n*(n-1)//2
    cons = np.empty(m)
    idx = 0
    for i in range(n):
        cons[idx] = c[i, 0] - r[i]; idx += 1
        cons[idx] = 1.0 - c[i, 0] - r[i]; idx += 1
        cons[idx] = c[i, 1] - r[i]; idx += 1
        cons[idx] = 1.0 - c[i, 1] - r[i]; idx += 1
    for i in range(n):
        for j in range(i+1, n):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            cons[idx] = np.sqrt(dx*dx + dy*dy) - r[i] - r[j]
            idx += 1
    return cons

def run_packing():
    n = 26
    centers = []
    radii = []
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + i*0.2, 0.1 + j*0.2])
            radii.append(0.04)
    centers.append([0.5, 0.5])
    radii.append(0.01)
    
    centers = np.array(centers)
    radii = np.array(radii)
    
    np.random.seed(42)
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0, 1)] * (2 * n) + [(0, 1)] * n
    
    cons_dict = {'type': 'ineq', 'fun': compute_constraints}
    
    res = minimize(objective_func, x0, bounds=bounds, 
                   constraints=cons_dict, method='SLSQP', 
                   options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
    
    opt_c = res.x[:2*n].reshape(n, 2)
    opt_r = res.x[2*n:]
    opt_r = np.maximum(opt_r, 0.0)
    
    return opt_c, opt_r, np.sum(opt_r)
