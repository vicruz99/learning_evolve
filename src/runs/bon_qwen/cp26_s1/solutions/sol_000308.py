# sol_000308 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 39d28b7b) state=dbab2f3a sum of radii=2.591505 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
OVERLAP_INDICES = np.triu_indices(N_CIRCLES, k=1)

def objective(vars):
    return -np.sum(vars[2*N_CIRCLES:])

def get_constraints(vars):
    X = vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    R = vars[2*N_CIRCLES:]
    
    b1 = X[:, 0] - R
    b2 = 1.0 - (X[:, 0] + R)
    b3 = X[:, 1] - R
    b4 = 1.0 - (X[:, 1] + R)
    
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum = R[:, np.newaxis] + R[np.newaxis, :]
    o1 = dist_sq[OVERLAP_INDICES] - r_sum[OVERLAP_INDICES]**2
    
    return np.concatenate([b1, b2, b3, b4, o1])

def get_initial_guess(seed):
    np.random.seed(seed)
    centers = np.zeros((N_CIRCLES, 2))
    idx = 0
    step = 0.14
    for r_idx in range(5):
        for c_idx in range(6):
            if idx < N_CIRCLES:
                x = 0.12 + c_idx * step + (0.5 * step if r_idx % 2 == 1 else 0.0)
                y = 0.12 + r_idx * step * 0.866
                centers[idx] = [x, y]
                idx += 1
    radii = np.ones(N_CIRCLES) * 0.04
    centers += np.random.randn(N_CIRCLES, 2) * 0.01
    radii += np.random.randn(N_CIRCLES) * 0.005
    radii = np.maximum(radii, 0.02)
    return np.hstack([centers.ravel(), radii])

def run_packing():
    best_val = -np.inf
    best_vars = None
    bounds = [(0.0, 1.0)] * (2*N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    for s in range(8):
        x0 = get_initial_guess(s)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            if res.success and -res.fun > best_val:
                best_val = -res.fun
                best_vars = res.x
        except:
            continue
            
    if best_vars is None:
        best_vars = get_initial_guess(0)
        
    centers = best_vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = best_vars[2*N_CIRCLES:]
    radii = np.maximum(radii, 1e-9)
    return centers, radii, float(np.sum(radii))
