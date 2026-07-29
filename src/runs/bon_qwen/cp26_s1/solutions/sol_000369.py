# sol_000369 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b90f636d) state=60ce214f sum of radii=1.950000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(params):
    """Objective: maximize sum of radii (minimize negative sum)"""
    return -np.sum(params[2::3])

def constraints(params):
    """Inequality constraints: boundaries and non-overlap"""
    n = N_CIRCLES
    # params layout: [x1, y1, r1, x2, y2, r2, ...]
    centers = params.reshape((n, 3))[:, :2]
    radii = params[2::3]

    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_boundary = np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])

    # Separation constraints: ||c_i - c_j||^2 >= (r_i + r_j)^2
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    rad_sum_sq = (radii[:, None] + radii[None, :])**2

    # Lower triangle indices to avoid duplicates and self-comparison
    i, j = np.tril_indices(n, -1)
    c_sep = dist_sq[i, j] - rad_sum_sq[i, j]

    return np.concatenate([c_boundary, c_sep])

def run_packing():
    n = N_CIRCLES
    
    # 1. Initialize with a hexagonal lattice pattern
    r_init = 0.075
    centers = np.zeros((n, 2))
    radii = np.full(n, r_init)
    
    # Row structure optimized for square container: 6-5-6-5-4 = 26 circles
    row_counts = [6, 5, 6, 5, 4]
    y = r_init
    idx = 0
    for row_i, count in enumerate(row_counts):
        x_start = r_init + (row_i % 2) * r_init
        for j in range(count):
            centers[idx, 0] = x_start + j * 2 * r_init
            centers[idx, 1] = y
            idx += 1
        y += np.sqrt(3) * r_init

    # Flatten to [x1, y1, r1, x2, y2, r2, ...]
    params = np.zeros(3 * n)
    params[0::3] = centers[:, 0]
    params[1::3] = centers[:, 1]
    params[2::3] = radii

    # Variable bounds
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # Constraint definition
    cons = {'type': 'ineq', 'fun': constraints}

    # 2. Optimize
    res = minimize(objective, params, method='SLSQP',
                   bounds=bounds, constraints=cons,
                   options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})

    # 3. Extract and return results
    final_params = res.x if res.success else params
    final_centers = np.column_stack((final_params[0::3], final_params[1::3]))
    final_radii = final_params[2::3]
    
    return final_centers, final_radii, np.sum(final_radii)
