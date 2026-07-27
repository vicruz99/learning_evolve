# sol_000054 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b794a107) state=aa109982 sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(v):
    """Minimize negative sum of radii"""
    return -np.sum(v[2 * N_CIRCLES:])

def boundary_con(v):
    """Vectorized boundary constraints: 4 per circle"""
    c = np.zeros(4 * N_CIRCLES)
    centers = v[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    r = v[2 * N_CIRCLES:]
    c[0::4] = centers[:, 0] - r
    c[1::4] = 1.0 - centers[:, 0] - r
    c[2::4] = centers[:, 1] - r
    c[3::4] = 1.0 - centers[:, 1] - r
    return c

def distance_con(v):
    """Vectorized non-overlap constraints: (N choose 2) constraints"""
    centers = v[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    r = v[2 * N_CIRCLES:]
    
    # Pairwise squared distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Squared sum of radii
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Constraint: dist_sq >= (r_i + r_j)^2
    con_mat = dist_sq - r_sum ** 2
    
    # Extract upper triangular indices (i < j)
    idx = np.triu_indices(N_CIRCLES, k=1)
    return con_mat[idx]

def run_packing():
    # 1. Hexagonal initial configuration
    r_init = 0.075
    centers = []
    # Row counts summing to 26
    rows = [6, 5, 6, 5, 4]
    y = r_init
    for k, count in enumerate(rows):
        for j in range(count):
            x = r_init + j * 2 * r_init
            if k % 2 == 1:
                x += r_init  # Shift odd rows for hexagonal packing
            centers.append([x, y])
        y += r_init * np.sqrt(3)

    centers = np.array(centers)
    radii = np.full(N_CIRCLES, r_init)

    # Flatten initial guess
    x0 = np.concatenate([centers.ravel(), radii])
    
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0, 1) for _ in range(2 * N_CIRCLES)] + [(0, 0.5) for _ in range(N_CIRCLES)]

    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': boundary_con},
        {'type': 'ineq', 'fun': distance_con}
    ]

    # 2. Optimize
    res = minimize(
        objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-11, 'disp': False}
    )

    # 3. Extract and post-process results
    best_centers = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    best_radii = res.x[2 * N_CIRCLES:]

    # Enforce strict validity
    best_radii = np.maximum(best_radii, 0.0)
    for i in range(N_CIRCLES):
        best_centers[i, 0] = np.clip(best_centers[i, 0], best_radii[i], 1.0 - best_radii[i])
        best_centers[i, 1] = np.clip(best_centers[i, 1], best_radii[i], 1.0 - best_radii[i])

    total_radius = float(np.sum(best_radii))
    return best_centers, best_radii, total_radius
