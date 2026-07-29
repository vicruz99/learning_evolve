# sol_000122 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 840b35ba) state=cdded62a sum of radii=2.503404 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(x):
    """Objective function to maximize the shared radius r (minimize -r)"""
    return -x[-1]

def compute_constraints(x, n):
    """
    Compute constraint violations.
    Returns an array where all elements should be >= 0.
    Constraints:
    1. Boundary: x_i >= r, 1-x_i >= r, y_i >= r, 1-y_i >= r
    2. Non-overlap: dist(c_i, c_j) >= 2r
    """
    r = x[-1]
    c = x[:-1].reshape(-1, 2)
    
    # Boundary constraints
    bnd = np.vstack([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ]).flatten()
    
    # Pairwise distance constraints
    # Compute differences vectorized
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    
    # Extract upper triangle indices to avoid duplicates and self-comparison
    i_idx, j_idx = np.triu_indices(n, k=1)
    non_overlap = dists[i_idx, j_idx] - 2.0 * r
    
    return np.concatenate([bnd, non_overlap])

def run_packing():
    n = 26
    
    # Bounds for variables: x_i, y_i in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)]
    
    # Initial configuration based on a hexagonal packing pattern
    # Row counts sum to 26: 6 + 5 + 6 + 5 + 4
    centers = np.zeros((n, 2))
    row_counts = [6, 5, 6, 5, 4]
    idx = 0
    
    for i, count in enumerate(row_counts):
        y = 0.15 + i * 0.1732
        # Shift odd rows to create hexagonal staggering
        start_x = 0.15 if i % 2 == 0 else 0.24
        xs = start_x + np.arange(count) * 0.19
        # Ensure x coordinates stay within reasonable bounds for initialization
        xs = xs[xs < 0.95]
        for x in xs:
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
                
    # Fill any remaining positions if row counts were trimmed
    while idx < n:
        centers[idx] = [0.5, 0.5] + 0.02 * np.random.randn(2)
        idx += 1
        
    # Concatenate centers and initial radius into a single vector
    x0 = np.concatenate([centers.flatten(), [0.04]])
    
    best_sol = None
    best_val = -np.inf
    
    # Run optimization multiple times with different seeds to avoid local minima
    for seed in range(5):
        np.random.seed(seed)
        x0_noisy = x0 + np.random.normal(0, 0.005, size=x0.shape)
        x0_noisy[:-1] = np.clip(x0_noisy[:-1], 0.0, 1.0)
        x0_noisy[-1] = np.clip(x0_noisy[-1], 0.0, 0.5)
        
        # Define constraint function without lambda
        def cons_fun(x):
            return compute_constraints(x, n)
            
        cons = {'type': 'ineq', 'fun': cons_fun}
        
        res = minimize(compute_objective, x0_noisy, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 3000, 'ftol': 1e-10})
        
        # Track the best solution found
        if -res.fun > best_val:
            best_val = -res.fun
            best_sol = res
            
    # Extract results
    final_centers = best_sol.x[:-1].reshape(-1, 2)
    final_r = best_sol.x[-1]
    
    # Apply a tiny safety margin to ensure strict non-overlap and boundary constraints
    # The validation allows 1e-12 tolerance, so 1e-7 is a safe conservative margin
    final_r = max(0.0, final_r - 1e-7)
    final_radii = np.full(n, final_r)
    
    return final_centers, final_radii, np.sum(final_radii)
