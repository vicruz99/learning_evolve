# sol_000326 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cc22fbce) state=db8cc182 sum of radii=2.614647 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def objective(params):
    """Objective function: minimize negative sum of radii."""
    n = len(params) // 3
    radii = params[2*n:]
    return -np.sum(radii)

def constraint_func(params):
    """
    Constraint function:
    - Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    - Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    Returns an array of constraint values >= 0.
    """
    n = len(params) // 3
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Boundary constraints
    c1 = centers[:, 0] - radii
    c2 = 1.0 - centers[:, 0] - radii
    c3 = centers[:, 1] - radii
    c4 = 1.0 - centers[:, 1] - radii
    
    # Pairwise constraints
    pair_c = []
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = np.sum((centers[i] - centers[j])**2)
            sum_r = radii[i] + radii[j]
            pair_c.append(dist_sq - sum_r**2)
            
    return np.concatenate([c1, c2, c3, c4, pair_c])

def get_initial_config(n):
    """Generates an initial feasible grid configuration."""
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.02)
    idx = 0
    y = 0.1
    while y <= 0.9 and idx < n:
        x = 0.1
        while x <= 0.9 and idx < n:
            centers[idx] = [x, y]
            x += 0.15
            idx += 1
        y += 0.15
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    n = 26
    x0 = get_initial_config(n)
    
    # Add small perturbation to avoid symmetry traps
    np.random.seed(42)
    x0 = x0 + np.random.randn(len(x0)) * 1e-3
    x0[:2*n] = np.clip(x0[:2*n], 0.05, 0.95)
    x0[2*n:] = np.clip(x0[2*n:], 0.02, 0.4)

    bounds = [(0, 1)] * (2*n) + [(0, 1)] * n
    cons = NonlinearConstraint(constraint_func, 0, np.inf)
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                   
    params = res.x
    centers = params[:2*n].reshape(n, 2)
    radii = np.maximum(params[2*n:], 0.0)
    
    return centers, radii, np.sum(radii)
