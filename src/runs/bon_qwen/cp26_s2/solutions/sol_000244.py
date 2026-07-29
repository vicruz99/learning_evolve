# sol_000244 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8f88c46f) state=9677aca4 sum of radii=2.626930 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_centers_from_params(params, N):
    """Transform parameters [u, v, r] to physical centers and radii."""
    u = params[0::3]
    v = params[1::3]
    r = params[2::3]
    # Automatic boundary enforcement: x = r + (1-2r)*u ensures x in [r, 1-r]
    x = r + (1.0 - 2.0 * r) * u
    y = r + (1.0 - 2.0 * r) * v
    return np.column_stack([x, y]), r

def objective_vec(params, N):
    """Negative sum of radii (we minimize this to maximize sum)."""
    _, r = get_centers_from_params(params, N)
    return -np.sum(r)

def constraints_vec(params, N):
    """Vectorized non-overlap constraints: dist_ij >= r_i + r_j."""
    centers, r = get_centers_from_params(params, N)
    # Compute pairwise distance matrix efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff**2, axis=2))
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Extract upper triangle indices to avoid duplicates and self-pairs
    i, j = np.triu_indices(N, k=1)
    return dist_mat[i, j] - r_sum[i, j]

def run_packing():
    N = 26
    u0 = np.zeros(3 * N)
    r_init = 0.04
    
    # Hexagonal lattice initialization for fast convergence
    dy = np.sqrt(3.0)/2.0 * 0.2
    ys = [0.12 + k*dy for k in range(5)]
    xs_even = [0.12 + k*0.2 for k in range(5)]
    xs_odd = [0.22 + k*0.2 for k in range(5)]
    
    idx = 0
    for k, y in enumerate(ys):
        xs = xs_even if k % 2 == 0 else xs_odd
        for x in xs:
            if idx < N:
                u0[3*idx] = x
                u0[3*idx+1] = y
                u0[3*idx+2] = r_init
                idx += 1
    # Fill any remaining slots if N doesn't match grid size exactly
    while idx < N:
        u0[3*idx] = 0.5
        u0[3*idx+1] = 0.5
        u0[3*idx+2] = r_init
        idx += 1
        
    # Bounds: u, v in [0, 1]; r in [1e-6, 0.49]
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.49)] * N
    
    cons = {'type': 'ineq', 'fun': constraints_vec, 'args': (N,)}
    
    res = minimize(
        objective_vec, 
        u0, 
        args=(N,), 
        bounds=bounds, 
        constraints=cons, 
        method='SLSQP', 
        options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False}
    )
                   
    centers, radii = get_centers_from_params(res.x, N)
    return centers, radii, float(np.sum(radii))
