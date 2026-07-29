# sol_000332 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 488bfafc) state=43ead8d5 sum of radii=2.587886 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x, n):
    radii = x[2*n:]
    return -np.sum(radii)

def overlap_constraints(x, n):
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    m = (n * (n - 1)) // 2
    vals = np.empty(m)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            vals[idx] = dist_sq - (radii[i] + radii[j]) ** 2
            idx += 1
    return vals

def boundary_constraints(x, n):
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    vals = np.empty(4 * n)
    for i in range(n):
        vals[4 * i] = centers[i, 0] - radii[i]
        vals[4 * i + 1] = 1.0 - centers[i, 0] - radii[i]
        vals[4 * i + 2] = centers[i, 1] - radii[i]
        vals[4 * i + 3] = 1.0 - centers[i, 1] - radii[i]
    return vals

def run_packing():
    n = 26
    # Initialize with a 5x5 grid + 1 circle at center
    r_init = 0.1
    centers_init = []
    radii_init = []
    for i in range(5):
        for j in range(5):
            centers_init.append([0.1 + i * 0.2, 0.1 + j * 0.2])
            radii_init.append(r_init)
    centers_init.append([0.5, 0.5])
    radii_init.append(r_init)
    
    x0 = np.array(centers_init).flatten()
    x0 = np.concatenate([x0, radii_init])
    
    # Break symmetry slightly to help optimizer escape local minima
    np.random.seed(42)
    x0 += np.random.uniform(-0.005, 0.005, size=x0.shape)
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-5, 0.5)] * n
    
    cons = [
        {'type': 'ineq', 'fun': overlap_constraints, 'args': (n,)},
        {'type': 'ineq', 'fun': boundary_constraints, 'args': (n,)}
    ]
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   args=(n,), options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
    
    centers = res.x[:2*n].reshape(n, 2)
    radii = res.x[2*n:]
    
    return centers, radii, float(np.sum(radii))
