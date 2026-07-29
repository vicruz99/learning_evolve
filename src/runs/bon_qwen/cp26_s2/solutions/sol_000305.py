# sol_000305 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5f579b4b) state=6193c2b9 sum of radii=2.592320 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(params, n_circles):
    """
    Computes all inequality constraints for the circle packing problem.
    Returns an array where each element must be >= 0.
    """
    centers = params[:2 * n_circles].reshape(n_circles, 2)
    radii = params[2 * n_circles:]

    # Pairwise non-overlap constraints: dist_ij >= r_i + r_j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    i_idx, j_idx = np.tril_indices(n_circles, -1)
    pairwise_cons = dists[i_idx, j_idx] - (radii[i_idx] + radii[j_idx])

    # Boundary constraints: r_i <= x_i <= 1-r_i  and  r_i <= y_i <= 1-r_i
    boundary_cons = np.concatenate([
        centers[:, 0] - radii,
        (1.0 - centers[:, 0]) - radii,
        centers[:, 1] - radii,
        (1.0 - centers[:, 1]) - radii
    ])

    return np.concatenate([pairwise_cons, boundary_cons])

def objective_function(params, n_circles):
    """
    Objective to minimize: negative sum of radii.
    """
    return -np.sum(params[2 * n_circles:])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    bounds = [(0.0, 1.0)] * (2 * n_circles) + [(0.0, 0.5)] * n_circles

    best_sum = -np.inf
    best_params = None

    # Run optimization from multiple random starting points
    for seed in range(15):
        np.random.seed(seed)
        # Initialize centers randomly within a safe margin, radii small
        centers = np.random.rand(n_circles, 2) * 0.8 + 0.1
        radii = np.full(n_circles, 0.02)
        x0 = np.concatenate([centers.flatten(), radii])

        constraint_def = {
            'type': 'ineq',
            'fun': compute_constraints,
            'args': (n_circles,)
        }

        res = minimize(
            fun=objective_function,
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_def,
            args=(n_circles,),
            options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
        )

        current_sum = np.sum(res.x[2 * n_circles:])
        if current_sum > best_sum:
            best_sum = current_sum
            best_params = res.x.copy()

    centers = best_params[:2 * n_circles].reshape(n_circles, 2)
    radii = best_params[2 * n_circles:]

    # Apply a minimal safety scaling to guarantee strict feasibility
    # against numerical tolerance in the validator
    radii *= 0.99999

    return centers, radii, np.sum(radii)
