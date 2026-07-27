# sol_000137 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4c86e033) state=2cec6d39 sum of radii=2.608466 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Precompute indices for pairwise constraints to avoid repeated allocation
TRIU_IDX = np.triu_indices(26, k=1)

def compute_constraints(vars):
    """
    Computes all constraint values as a single 1D array.
    All values must be >= 0 for feasibility.
    """
    n = 26
    centers = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c1 = centers[:, 0] - r
    c2 = 1.0 - centers[:, 0] - r
    c3 = centers[:, 1] - r
    c4 = 1.0 - centers[:, 1] - r
    
    # Pairwise distance constraints: dist^2 >= (r_i + r_j)^2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    c5 = dist_sq[TRIU_IDX] - r_sum[TRIU_IDX]**2
    
    return np.concatenate([c1, c2, c3, c4, c5])

def objective(vars):
    """
    Objective function: minimize negative sum of radii.
    """
    n = len(vars) // 3
    return -np.sum(vars[2*n:])

def run_packing():
    n = 26
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons = [{'type': 'ineq', 'fun': compute_constraints}]
    
    # Multi-start optimization to avoid local minima
    for seed in range(5):
        rng = np.random.RandomState(seed)
        
        # Initial grid layout
        x = np.linspace(0.1, 0.9, 6)
        y = np.linspace(0.1, 0.9, 5)
        cx, cy = np.meshgrid(x, y)
        centers_init = np.column_stack([cx.ravel(), cy.ravel()])[:n]
        
        # Perturb centers and initialize radii
        centers_init += rng.uniform(-0.02, 0.02, centers_init.shape)
        radii_init = 0.05 * np.ones(n) + rng.uniform(-0.005, 0.005, n)
        
        x0 = np.concatenate([centers_init.flatten(), radii_init])
        
        # Run optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                      constraints=cons, options={'maxiter': 3000})
                      
        if not np.isnan(res.fun):
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = res.x[:2*n].reshape(n, 2)
                best_radii = res.x[2*n:]
                
    # Final result preparation
    if best_centers is not None:
        # Slight shrinkage guarantees strict satisfaction of 1e-12 tolerance
        best_radii = best_radii * 0.999
    else:
        # Fallback configuration (should not be reached in practice)
        best_centers = np.tile([0.5, 0.5], (n, 1))
        best_radii = np.zeros(n)
        
    return best_centers, best_radii, np.sum(best_radii)
