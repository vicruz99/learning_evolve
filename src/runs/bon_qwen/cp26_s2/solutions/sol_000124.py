# sol_000124 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 840b35ba) state=1f2c7903 sum of radii=2.627565 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_func(z):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(z[2::3])

def constraint_func(z):
    """Inequality constraints: boundaries and non-overlap."""
    n = 26
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    c = []
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c.extend(x - r)
    c.extend(1.0 - x - r)
    c.extend(y - r)
    c.extend(1.0 - y - r)
    
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            c.append(dx*dx + dy*dy - dr*dr)
            
    return np.array(c)

def run_packing():
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -1.0

    # Initial hexagonal-like grid placement
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(5):
        y = row * 0.2 + 0.1
        shift = 0.1 if row % 2 == 1 else 0.0
        for col in range(5):
            if idx >= n:
                break
            x = col * 0.2 + shift + 0.1
            centers[idx] = [x, y]
            idx += 1
    while idx < n:
        centers[idx] = [0.5, 0.5 + (idx - 25) * 0.15]
        idx += 1

    radii_init = np.full(n, 0.085)
    
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0) for _ in range(3 * n)]
    for i in range(n):
        bounds[2 + 3 * i] = (0.0, 0.5)

    cons = {'type': 'ineq', 'fun': constraint_func}

    # Multiple restarts with different seeds to avoid local optima
    for seed in range(5):
        rng = np.random.RandomState(seed)
        z0 = np.zeros(3 * n)
        for i in range(n):
            z0[3 * i] = np.clip(centers[i, 0] + rng.uniform(-0.02, 0.02), 0.05, 0.95)
            z0[3 * i + 1] = np.clip(centers[i, 1] + rng.uniform(-0.02, 0.02), 0.05, 0.95)
            z0[3 * i + 2] = np.clip(radii_init[i] + rng.uniform(-0.005, 0.005), 0.01, 0.4)

        res = minimize(obj_func, z0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 5000, 'ftol': 1e-12})

        current_sum = np.sum(res.x[2::3])
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
            best_radii = res.x[2::3]

    # Numerical safety: ensure constraints are satisfied within tolerance
    z_check = np.zeros(3 * n)
    for i in range(n):
        z_check[3 * i] = best_centers[i, 0]
        z_check[3 * i + 1] = best_centers[i, 1]
        z_check[3 * i + 2] = best_radii[i]

    c_vals = constraint_func(z_check)
    min_c = np.min(c_vals)
    if min_c < -1e-8:
        # Slight shrinkage to guarantee validity
        best_radii = best_radii * 0.995
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum
