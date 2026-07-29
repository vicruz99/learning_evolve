# sol_000106 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f3700082) state=422de027 sum of radii=2.575397 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _init_packing(n):
    """Initialize circle positions using a hexagonal lattice pattern."""
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.085
    r = 0.085
    sqrt3 = np.sqrt(3)
    idx = 0
    y = r
    # Hexagonal pattern row counts summing to 26
    counts = [5, 4, 5, 4, 5, 3]
    
    for row, cnt in enumerate(counts):
        x_offset = r if row % 2 == 1 else 0.0
        for i in range(cnt):
            if idx < n:
                centers[idx, 0] = r + x_offset + i * 2 * r
                centers[idx, 1] = y
                idx += 1
        y += r * sqrt3
    return centers, radii

def _objective(x, n):
    """Objective function: maximize sum of radii."""
    return -np.sum(x[2*n:])

def _constraints(x, n):
    """Constraint function: non-overlap and boundary conditions."""
    pts = x[:2*n].reshape(n, 2)
    rs = x[2*n:]
    
    # Vectorized pairwise squared distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Upper triangle indices for i < j
    triu_idx = np.triu_indices(n, k=1)
    overlap_c = dist_sq[triu_idx] - (rs[:, None] + rs[None, :])[triu_idx]**2
    
    # Boundary constraints: r <= x,y <= 1-r
    bound_c = np.concatenate([
        pts[:, 0] - rs,
        1.0 - pts[:, 0] - rs,
        pts[:, 1] - rs,
        1.0 - pts[:, 1] - rs
    ])
    
    return np.concatenate([overlap_c, bound_c])

def run_packing():
    n = 26
    centers0, radii0 = _init_packing(n)
    x0 = np.hstack([centers0.flatten(), radii0])
    
    bounds = [(0.0, 1.0)] * (2*n) + [(1e-7, 0.5)] * n
    
    best_res = None
    best_sum = -np.inf
    
    # Multiple restarts to avoid local minima
    for _ in range(5):
        x_trial = x0 + np.random.normal(0, 0.005, size=x0.shape)
        x_trial[:2*n] = np.clip(x_trial[:2*n], 0.0, 1.0)
        x_trial[2*n:] = np.clip(x_trial[2*n:], 1e-7, 0.5)
        
        res = minimize(_objective, x_trial, args=(n,), method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': _constraints, 'args': (n,)},
                       options={'maxiter': 5000, 'ftol': 1e-10})
        if res.success:
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_res = res
                
    if best_res is not None:
        opt_c = best_res.x[:2*n].reshape(n, 2)
        opt_r = best_res.x[2*n:]
        return opt_c, opt_r, np.sum(opt_r)
    else:
        return centers0, radii0, np.sum(radii0)
