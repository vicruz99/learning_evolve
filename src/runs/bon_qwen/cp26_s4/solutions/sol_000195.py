# sol_000195 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bafdbd7e) state=7f43cb5d sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_min_separation(coords):
    """Compute the minimum distance between any two circles or between a circle and the boundary."""
    n = coords.shape[0]
    min_sep = 2.0
    for i in range(n):
        x, y = coords[i]
        min_sep = min(min_sep, x, 1-x, y, 1-y)
        for j in range(i+1, n):
            dx, dy = coords[i] - coords[j]
            d = np.sqrt(dx*dx + dy*dy)
            min_sep = min(min_sep, d)
    return min_sep

def objective(vars):
    """Objective: maximize t (minimum separation distance). Return negative for minimization."""
    return -vars[2*N_CIRCLES]

def bound_x_min(vars):
    """Constraint: x_i >= t/2  =>  x_i - t/2 >= 0"""
    return vars[::2] - vars[2*N_CIRCLES]/2.0

def bound_x_max(vars):
    """Constraint: x_i <= 1 - t/2  =>  t/2 - 1 + x_i >= 0"""
    return vars[2*N_CIRCLES]/2.0 - 1.0 + vars[::2]

def bound_y_min(vars):
    """Constraint: y_i >= t/2  =>  y_i - t/2 >= 0"""
    return vars[1::2] - vars[2*N_CIRCLES]/2.0

def bound_y_max(vars):
    """Constraint: y_i <= 1 - t/2  =>  t/2 - 1 + y_i >= 0"""
    return vars[2*N_CIRCLES]/2.0 - 1.0 + vars[1::2]

def pair_dists(vars):
    """Constraint: dist(i,j) >= t  =>  dist_sq(i,j) - t^2 >= 0"""
    coords = vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    t = vars[2*N_CIRCLES]
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    d2 = np.sum(diff**2, axis=2)
    triu_indices = np.triu_indices(N_CIRCLES, k=1)
    return d2[triu_indices] - t*t

def run_packing():
    constraints = [
        {'type': 'ineq', 'fun': bound_x_min},
        {'type': 'ineq', 'fun': bound_x_max},
        {'type': 'ineq', 'fun': bound_y_min},
        {'type': 'ineq', 'fun': bound_y_max},
        {'type': 'ineq', 'fun': pair_dists}
    ]
    bounds = [(0, 1)] * (2 * N_CIRCLES) + [(0, 1)]

    best_t = 0.0
    best_centers = None

    # Multiple random restarts to escape local minima
    seeds = [42, 123, 456, 789, 1024, 2048, 3141]
    for seed in seeds:
        np.random.seed(seed)
        # Initialize circles in the center region to avoid immediate boundary penalties
        centers_init = np.random.rand(N_CIRCLES, 2) * 0.6 + 0.2
        t_init = 0.05
        x0 = np.concatenate([centers_init.flatten(), [t_init]])

        res = minimize(objective, x0, method='SLSQP', constraints=constraints,
                       bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})

        # Keep the best result found
        if -res.fun > best_t:
            best_t = -res.fun
            best_centers = res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)

    # Compute exact feasible radius from the optimized configuration
    min_sep = compute_min_separation(best_centers)
    # Safety factor ensures strict compliance with validation tolerances
    r_final = min_sep / 2.0 * 0.9999
    radii = np.full(N_CIRCLES, r_final)
    
    return best_centers, radii, np.sum(radii)
