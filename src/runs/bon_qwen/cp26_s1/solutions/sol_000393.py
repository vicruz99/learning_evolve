# sol_000393 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 19a68663) state=a2615161 sum of radii=2.462463 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def constr_pair(vars, n, indices):
    c = vars[:-1].reshape(n, 2)
    t = vars[-1]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    i, j = indices
    return d[i, j] - t

def constr_bound(vars, n):
    c = vars[:-1].reshape(n, 2)
    t = vars[-1]
    ht = t / 2.0
    return np.concatenate([c[:,0]-ht, 1.0-c[:,0]-ht, c[:,1]-ht, 1.0-c[:,1]-ht])

def constr_t(vars):
    return vars[-1]

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    r_idx = 0
    sx = 0.20
    sy = sx * np.sqrt(3) / 2.0
    
    # Generate hexagonal initial positions
    while idx < n:
        cols = 5 if r_idx < 5 else 1
        c_idx = 0
        while c_idx < cols and idx < n:
            x = 0.2 + c_idx * sx
            y = 0.2 + r_idx * sy
            if r_idx % 2 == 1:
                x += sx / 2.0
            centers[idx] = [x, y]
            idx += 1
            c_idx += 1
        r_idx += 1
        
    # Center and scale to fit comfortably inside [0,1]
    cx, cy = centers.mean(axis=0)
    centers -= np.array([cx, cy])
    scale = 1.0 / (centers.max() - centers.min()) * 0.8
    centers *= scale
    centers += 0.5
    
    # Precompute constraint indices
    indices = np.tril_indices(n, -1)
    
    # Compute initial feasible t
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    i, j = indices
    min_dist = np.min(dists[i, j])
    bnd = np.min(np.array([centers[:,0], 1.0-centers[:,0], centers[:,1], 1.0-centers[:,1]]))
    t0 = 0.95 * min(min_dist, 2.0 * bnd)
    
    x0 = np.concatenate([centers.flatten(), [t0]])
    
    cons = [
        {'type': 'ineq', 'fun': constr_pair, 'args': (n, indices)},
        {'type': 'ineq', 'fun': constr_bound, 'args': (n,)},
        {'type': 'ineq', 'fun': constr_t}
    ]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 1.0)]
    
    try:
        res = minimize(lambda v: -v[-1], x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 5000, 'ftol': 1e-10})
        opt_centers = res.x[:-1].reshape(n, 2)
        opt_t = max(res.x[-1], 1e-6)
    except Exception:
        opt_centers = centers
        opt_t = t0
        
    opt_centers = np.clip(opt_centers, 0.0, 1.0)
    radii = np.full(n, opt_t / 2.0)
    
    # Slight shrinkage to guarantee validation passes despite floating point arithmetic
    radii *= 0.999999
    
    return opt_centers, radii, np.sum(radii)
