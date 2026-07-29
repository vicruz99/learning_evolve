# sol_000007 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 92133c71) state=d2fd70ea sum of radii=2.445979 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    N = 26
    
    # Helper functions for constraints (top-level to avoid closures)
    def bound_constraints(v, N):
        r = v[-1]
        c = v[:-1].reshape((N, 2))
        vals = np.empty(4 * N)
        vals[0::4] = c[:, 0] - r          # x >= r
        vals[1::4] = 1.0 - r - c[:, 0]    # x <= 1-r
        vals[2::4] = c[:, 1] - r          # y >= r
        vals[3::4] = 1.0 - r - c[:, 1]    # y <= 1-r
        return vals

    def pair_constraints(v, N):
        r = v[-1]
        c = v[:-1].reshape((N, 2))
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists_sq = np.sum(diffs**2, axis=2)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        return dists_sq[mask] - 4 * (r**2)

    # 1. Initialization: Hexagonal packing
    r_init = 0.105
    centers0 = []
    y = r_init
    row = 0
    while len(centers0) < N:
        x_off = (row % 2) * r_init
        x = r_init + x_off
        while x < 1.0 - r_init + 1e-6 and len(centers0) < N:
            centers0.append([x, y])
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row += 1
    centers0 = np.array(centers0[:N])
    
    # Variable vector: [x0, y0, x1, y1, ..., x25, y25, r]
    x0 = np.concatenate([centers0.flatten(), [r_init]])
    
    # Bounds: coordinates in [0, 1], radius in [0.01, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.01, 0.5)]
    
    # Constraints setup
    cons = [
        {'type': 'ineq', 'fun': lambda v: bound_constraints(v, N)},
        {'type': 'ineq', 'fun': lambda v: pair_constraints(v, N)}
    ]
    
    # Objective: maximize r  <=>  minimize -r
    def obj(v):
        return -v[-1]
        
    # 2. Optimization
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 500, 'ftol': 1e-12})
    
    if res.success:
        centers_opt = res.x[:2*N].reshape((N, 2))
        r_opt = res.x[-1]
    else:
        # Fallback to initial configuration if optimizer fails
        centers_opt = centers0
        r_opt = r_init
        
    # 3. Final Validation & Safety Adjustment
    # Compute the true maximum feasible radius from the optimized positions
    r_bound = 1.0
    r_pair = 1.0
    
    for i in range(N):
        x, y = centers_opt[i]
        r_bound = min(r_bound, x, 1.0 - x, y, 1.0 - y)
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
            r_pair = min(r_pair, d / 2.0)
            
    # Take the minimum and apply a small safety factor for numerical stability
    r_final = min(r_bound, r_pair) * 0.9995
    radii = np.full(N, r_final)
    
    return centers_opt, radii, np.sum(radii)
