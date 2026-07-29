# sol_000161 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a0a8497a) state=78daf5fb sum of radii=2.166667 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_obj(vars):
    # Objective: maximize radius r (minimize -r)
    return -vars[-1]

def get_bounds_constraints(vars):
    n = 26
    # Extract positions and radius
    x = vars[:n*2].reshape(n, 2)[:, 0]
    y = vars[:n*2].reshape(n, 2)[:, 1]
    r = vars[-1]
    
    # Constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # Equivalent to: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c = np.empty(4*n)
    c[:n] = x - r
    c[n:2*n] = 1.0 - x - r
    c[2*n:3*n] = y - r
    c[3*n:] = 1.0 - y - r
    return c

def get_sep_constraints(vars):
    n = 26
    pos = vars[:n*2].reshape(n, 2)
    r = vars[-1]
    
    # Compute pairwise distances efficiently
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Upper triangular indices for i < j
    iu, ju = np.triu_indices(n, k=1)
    # Constraint: dist >= 2*r  =>  dist - 2*r >= 0
    return dists[iu, ju] - 2.0 * r

def run_packing():
    n = 26
    
    # Initial guess: Hexagonal packing layout scaled to fit in [0,1]^2
    r_init = 0.10
    pos_list = []
    # 5 rows with alternating 6 and 4 circles: 6+4+6+4+6 = 26
    for row in range(5):
        y = row * np.sqrt(3) * r_init
        if row % 2 == 0:
            cols = 6
            x_off = r_init
        else:
            cols = 4
            x_off = 3 * r_init
        for c in range(cols):
            pos_list.append([x_off + c * 2 * r_init, y])

    pos = np.array(pos_list)
    # Normalize to [0.1, 0.9] to start safely inside the unit square
    pos = (pos - pos.min(0)) / (pos.max(0) - pos.min(0)) * 0.8 + 0.1
    
    # Variables: [x1, y1, ..., x26, y26, r]
    x0 = np.concatenate([pos.flatten(), [r_init]])

    bounds = [(0.0, 1.0)] * (n*2) + [(0.0, 0.5)]
    cons = [
        {'type': 'ineq', 'fun': get_bounds_constraints},
        {'type': 'ineq', 'fun': get_sep_constraints}
    ]

    # Run SLSQP optimization
    res = minimize(get_obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12})

    centers = res.x[:n*2].reshape(n, 2)
    r_opt = res.x[-1]
    radii = np.full(n, r_opt)

    # Clamp centers to ensure strict boundary validity (handles numerical drift)
    centers = np.clip(centers, r_opt, 1.0 - r_opt)

    total_sum = float(np.sum(radii))
    return centers, radii, total_sum
