# sol_000100 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bf51a1cd) state=64ec09e3 sum of radii=2.166667 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective function: maximize radius r (minimize -r)"""
    return -vars[-1]

def constraint_func(vars):
    """
    Returns inequality constraints g(vars) >= 0:
    1. Boundary constraints: circles inside [0,1]^2
    2. Distance constraints: no overlaps
    """
    r = vars[-1]
    c = vars[:-1].reshape(N_CIRCLES, 2)
    
    # Boundary constraints
    con_b = np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])
    
    # Distance constraints: ||c_i - c_j||^2 >= 4r^2
    cx = c[:, 0]
    cy = c[:, 1]
    dx = cx[:, np.newaxis] - cx
    dy = cy[:, np.newaxis] - cy
    d2 = dx**2 + dy**2
    
    idx = np.triu_indices(N_CIRCLES, k=1)
    con_d = d2[idx] - 4.0 * r * r
    
    return np.concatenate([con_b, con_d])

def run_packing():
    # 1. Initialize centers on a hexagonal lattice
    r0 = 0.08
    centers = []
    y = r0
    row = 0
    while len(centers) < N_CIRCLES:
        x = r0 if row % 2 == 0 else 2.0 * r0
        while x + r0 <= 1.0 and len(centers) < N_CIRCLES:
            centers.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
        row += 1
        
    centers = np.array(centers[:N_CIRCLES])
    x0 = np.concatenate([centers.flatten(), [r0]])
    
    # 2. Setup optimization problem
    cons = {'type': 'ineq', 'fun': constraint_func}
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)]
    
    # 3. Run SLSQP optimizer
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons,
        options={'ftol': 1e-12, 'maxiter': 5000, 'disp': False}
    )
    
    # 4. Extract results
    opt_r = max(res.x[-1], 0.0)
    opt_centers = res.x[:-1].reshape(N_CIRCLES, 2)
    
    radii = np.full(N_CIRCLES, opt_r)
    return opt_centers, radii, np.sum(radii)
