# sol_000020 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1c033854) state=9b156014 sum of radii=2.604992 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_constraints(vars):
    n = N_CIRCLES
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    cons = np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])
    
    # Pairwise distance constraints: ||c_i - c_j|| >= r_i + r_j
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    pair_cons = dists - r_sum
    
    # Extract upper triangle to avoid duplicates and self-pairs
    triu_indices = np.triu_indices(n, k=1)
    pair_cons = pair_cons[triu_indices]
    
    return np.concatenate([cons, pair_cons])

def objective(vars):
    n = N_CIRCLES
    r = vars[2*n:]
    return -np.sum(r)

def run_packing():
    n = N_CIRCLES
    
    # Initial placement: staggered hexagonal grid
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.095)
    
    idx = 0
    row_counts = [5, 6, 5, 6, 4]  # Sums to 26
    dy = 0.2
    for i, count in enumerate(row_counts):
        y = 0.1 + i * dy
        spacing = 1.0 / (count + 1)
        shift = 0.5 * spacing if i % 2 != 0 else 0.0
        for j in range(count):
            if idx < n:
                centers[idx, 0] = (j + 1) * spacing + shift
                centers[idx, 1] = y
                idx += 1
                
    # Ensure initial positions are strictly inside to avoid immediate boundary violations
    centers = np.clip(centers, 0.05, 0.95)
    x0 = np.concatenate([centers.flatten(), radii])
    
    cons = {'type': 'ineq', 'fun': compute_constraints}
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # SLSQP is effective for this class of smooth nonlinear constraints
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                   
    final_centers = res.x[:2*n].reshape(n, 2)
    final_radii = res.x[2*n:]
    
    # Clamp to strictly satisfy non-negativity
    final_radii = np.maximum(final_radii, 1e-9)
    
    return final_centers, final_radii, float(np.sum(final_radii))
