# sol_000264 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3433cac4) state=8ede8a9a sum of radii=2.597969 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
TRIL_IDX = np.tril_indices(N, k=-1)

def objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2*N:])

def constraint(x):
    """Inequality constraints: boundary and non-overlap."""
    c = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    # Boundary constraints: 4 per circle
    b = np.empty(4*N)
    b[0::4] = c[:, 0] - r
    b[1::4] = 1.0 - c[:, 0] - r
    b[2::4] = c[:, 1] - r
    b[3::4] = 1.0 - c[:, 1] - r
    
    # Pairwise non-overlap constraints: dist_sq >= (r_i + r_j)^2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    sum_r_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    # Extract lower triangle (i > j)
    p = dist_sq[TRIL_IDX] - sum_r_sq[TRIL_IDX]
    
    return np.concatenate([b, p])

def run_packing():
    cons = {'type': 'ineq', 'fun': constraint}
    bnds = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N
    
    best_sum = 0.0
    best_res = None
    
    # Multiple random restarts to find global optimum basin
    for seed in range(20):
        rng = np.random.RandomState(seed)
        # Start with small feasible circles
        c0 = rng.uniform(0.1, 0.9, (N, 2))
        r0 = 0.05 * np.ones(N)
        x0 = np.concatenate([c0.flatten(), r0])
        
        res = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons,
                       options={'maxiter': 1500, 'ftol': 1e-9, 'disp': False})
        
        current_sum = np.sum(res.x[2*N:])
        if current_sum > best_sum:
            # Verify constraints are satisfied within tolerance
            vals = constraint(res.x)
            if np.min(vals) >= -1e-6:
                best_sum = current_sum
                best_res = (res.x[:2*N].reshape(N, 2).copy(), res.x[2*N:].copy())
                
    # Fallback to a valid simple grid if optimization fails
    if best_res is None:
        c_fb = np.zeros((N, 2))
        r_fb = np.full(N, 0.05)
        for i in range(N):
            c_fb[i, 0] = (i % 5) * 0.2 + 0.1
            c_fb[i, 1] = (i // 5) * 0.2 + 0.1
        best_res = (c_fb, r_fb)
        best_sum = np.sum(r_fb)
        
    return best_res[0], best_res[1], best_sum
