# sol_000078 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ed1177e6) state=648c5630 sum of radii=2.610526 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
UPPER_INDICES = np.triu_indices(N, k=1)

def objective(x):
    return -np.sum(x[2*N:])

def constr_boundary(x):
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    c1 = centers[:, 0] - radii
    c2 = 1.0 - centers[:, 0] - radii
    c3 = centers[:, 1] - radii
    c4 = 1.0 - centers[:, 1] - radii
    return np.concatenate([c1, c2, c3, c4])

def constr_dist(x):
    centers = x[:2*N].reshape(N, 2)
    radii = x[2*N:]
    dist2 = np.sum((centers[:, None, :] - centers[None, :, :])**2, axis=2)
    r_sum2 = (radii[:, None] + radii[None, :])**2
    return (dist2 - r_sum2)[UPPER_INDICES]

def run_packing():
    centers = np.zeros((N, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
            idx += 1
    centers[idx] = [0.2, 0.2]
    
    radii = np.full(N, 0.09)
    radii[-1] = 0.04
    
    x0 = np.concatenate([centers.ravel(), radii])
    
    bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
    
    cons = [
        {'type': 'ineq', 'fun': constr_boundary},
        {'type': 'ineq', 'fun': constr_dist}
    ]
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False})
                   
    centers_out = res.x[:2*N].reshape(N, 2)
    radii_out = res.x[2*N:]
    radii_out = np.maximum(radii_out, 1e-9)
    
    return centers_out, radii_out, np.sum(radii_out)
