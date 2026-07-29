# sol_000232 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dc099519) state=e5da7695 sum of radii=2.614996 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
_TRI_IDX = np.triu_indices(N_CIRCLES, k=1)

def objective(vars):
    return -np.sum(vars[2*N_CIRCLES:])

def constraint_boundary(vars):
    xs = vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    rs = vars[2*N_CIRCLES:]
    return np.concatenate([
        xs[:, 0] - rs,
        1.0 - xs[:, 0] - rs,
        xs[:, 1] - rs,
        1.0 - xs[:, 1] - rs
    ])

def constraint_overlap(vars):
    xs = vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    rs = vars[2*N_CIRCLES:]
    diff = xs[:, None, :] - xs[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    rs_sum = rs[:, None] + rs[None, :]
    return dist_sq[_TRI_IDX] - rs_sum[_TRI_IDX]**2

def generate_initial_guess(seed):
    rng = np.random.default_rng(seed)
    pts = []
    # Hexagonal-ish grid layout
    for i in range(5):
        y = 0.12 + i * 0.17
        offset = 0.085 * (i % 2)
        for j in range(6):
            x = 0.12 + j * 0.15 + offset
            if len(pts) < N_CIRCLES:
                pts.append([x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02)])
    pts = np.array(pts[:N_CIRCLES])
    
    # Compute strictly feasible initial radii
    rs = np.zeros(N_CIRCLES)
    for i in range(N_CIRCLES):
        d_bound = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        diffs = pts[i] - pts
        d_pts = np.sqrt(np.sum(diffs**2, axis=1))
        d_pts[i] = np.inf
        min_dist = np.min(d_pts)
        rs[i] = min(d_bound, min_dist) * 0.45
        
    return np.concatenate([pts.flatten(), rs])

def run_packing():
    n = N_CIRCLES
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]

    best_x = None
    best_val = -np.inf

    # Multi-start to find good basin
    for seed in range(6):
        x0 = generate_initial_guess(seed)
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
        if -res.fun > best_val:
            best_val = -res.fun
            best_x = res.x

    # High-precision refinement
    res_final = minimize(objective, best_x, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
    
    centers = res_final.x[:2*n].reshape(n, 2)
    radii = res_final.x[2*n:]
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, np.sum(radii)
