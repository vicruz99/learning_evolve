# sol_000252 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 30e75f73) state=996bf809 sum of radii=2.554984 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def constraint_fun(params, n, iu, ju):
    """
    Evaluate inequality constraints: boundary containment and non-overlap.
    Returns array where all elements must be >= 0.
    """
    cs = params[:n*2].reshape(n, 2)
    rs = params[n*2:]
    
    # Boundary constraints
    b1 = cs[:, 0] - rs
    b2 = 1.0 - cs[:, 0] - rs
    b3 = cs[:, 1] - rs
    b4 = 1.0 - cs[:, 1] - rs
    
    # Overlap constraints (vectorized)
    diff = cs[:, np.newaxis, :] - cs[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    rad_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    ovl = dist_sq[iu, ju] - rad_sum[iu, ju]**2
    
    return np.concatenate([b1, b2, b3, b4, ovl])

def run_packing():
    n = 26
    iu, ju = np.triu_indices(n, k=1)
    
    # 1. Hexagonal lattice initialization for high density
    centers = np.zeros((n, 2))
    idx = 0
    row = 0
    r_est = 0.085
    h = r_est * np.sqrt(3)
    
    while idx < n:
        n_in_row = 6 if row % 2 == 0 else 5
        if idx + n_in_row > n:
            n_in_row = n - idx
        x_start = r_est + (row % 2) * r_est
        for i in range(n_in_row):
            if idx < n:
                centers[idx, 0] = x_start + i * 2 * r_est
                centers[idx, 1] = r_est + row * h
                idx += 1
        row += 1
        
    # 2. Conservative initial radii to ensure feasibility
    radii_init = np.full(n, 0.05)
    x0 = np.concatenate([centers.flatten(), radii_init])
    
    def objective(params):
        return -np.sum(params[n*2:])
        
    cons = {'type': 'ineq', 'fun': constraint_fun, 'args': (n, iu, ju)}
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # 3. SLSQP optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 3000, 'ftol': 1e-10})
    
    opt_centers = res.x[:n*2].reshape(n, 2)
    opt_radii = np.maximum(res.x[n*2:], 1e-7)
    
    return opt_centers, opt_radii, np.sum(opt_radii)
