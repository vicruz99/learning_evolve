# sol_000204 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 263f0241) state=fc677d8c sum of radii=2.525455 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def objective(vars, n):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    return -np.sum(vars[2*n:])

def constraints_vals(vars, n):
    """Compute all constraint values vectorized for speed."""
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    con_list = []
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    for i in range(n):
        con_list.extend([
            c[i, 0] - r[i],
            1.0 - c[i, 0] - r[i],
            c[i, 1] - r[i],
            1.0 - c[i, 1] - r[i]
        ])
        
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Broadcasting for efficient computation
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    pair_con = dist_sq - r_sum_sq
    
    # Extract upper triangle (i < j)
    triu_idx = np.triu_indices(n, k=1)
    con_list.extend(pair_con[triu_idx])
    
    return np.array(con_list)

def constraint_wrapper(vars):
    """Wrapper to fix n=26 for the optimizer."""
    return constraints_vals(vars, 26)

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Hexagonal lattice initialization for high packing density
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.02)  # Start feasible with small radii
    idx = 0
    row, col = 0, 0
    dx, dy = 0.24, 0.24 * np.sqrt(3)/2
    
    while idx < n:
        x = col * dx + (row % 2) * dx/2
        y = row * dy
        if x < 0.95 and y < 0.95:
            centers[idx] = [x, y]
            idx += 1
            col += 1
            if col * dx > 0.9:
                col = 0
                row += 1
        else:
            centers[idx] = [0.5, 0.5]
            idx += 1
            
    x0 = np.concatenate([centers.flatten(), radii])
    con = NonlinearConstraint(constraint_wrapper, 0.0, np.inf)
    
    # Optimize
    res = minimize(
        objective, 
        x0, 
        args=(n,), 
        method='trust-constr',
        bounds=bounds, 
        constraints=con,
        options={'maxiter': 200, 'verbose': 0, 'gtol': 1e-6}
    )
    
    # Extract and validate solution
    if res.success:
        best_c = res.x[:2*n].reshape((n, 2))
        best_r = res.x[2*n:]
    else:
        best_c = centers
        best_r = radii
        
    # Ensure strict validity (clip tiny negative drifts)
    best_r = np.clip(best_r, 1e-9, 0.5)
    best_c[:, 0] = np.clip(best_c[:, 0], best_r, 1.0 - best_r)
    best_c[:, 1] = np.clip(best_c[:, 1], best_r, 1.0 - best_r)
    
    final_sum = np.sum(best_r)
    return best_c, best_r, final_sum
