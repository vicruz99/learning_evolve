# sol_000304 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e3d19f45) state=72662f93 sum of radii=2.472649 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

# Global constant for number of circles
N_CIRCLES = 26

def generate_initial_centers(n):
    """
    Generates initial centers for n circles based on a dense hexagonal cluster.
    """
    # Generate a hexagonal lattice with spacing 2 (r=1)
    # Basis vectors: (2, 0) and (1, sqrt(3))
    # Points: (2*i + j, sqrt(3)*j)
    points = []
    # Range large enough to cover n points
    for i in range(-10, 11):
        for j in range(-10, 11):
            x = 2 * i + j
            y = math.sqrt(3) * j
            points.append([x, y])
    
    points = np.array(points)
    # Sort by distance from origin to get compact cluster
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Pick top n points
    initial_pts = points[sorted_indices[:n]]
    
    # Scale to fit in unit square
    min_x, min_y = np.min(initial_pts, axis=0)
    max_x, max_y = np.max(initial_pts, axis=0)
    
    # Required size including radius 1 on all sides
    width_req = (max_x - min_x) + 2.0
    height_req = (max_y - min_y) + 2.0
    
    # Scale factor to fit in 1x1
    scale = 1.0 / max(width_req, height_req)
    
    # Initial radius
    r_init = scale
    
    # Transform points: shift to 0, scale, add r_init to center
    pts_shifted = initial_pts - np.array([min_x, min_y])
    centers = pts_shifted * scale + r_init
    return centers, r_init

def objective(x, n=N_CIRCLES):
    """
    Objective function: maximize radius r (minimize -r).
    """
    return -x[-1]

def constraint_bounds(x, n=N_CIRCLES):
    """
    Boundary constraints: r <= x <= 1-r, r <= y <= 1-r.
    Returns array of inequalities g(x) >= 0.
    """
    r = x[-1]
    c = x[:-1].reshape(n, 2)
    return np.concatenate([
        c[:, 0] - r,
        1 - c[:, 0] - r,
        c[:, 1] - r,
        1 - c[:, 1] - r
    ])

def constraint_overlap(x, n=N_CIRCLES):
    """
    Non-overlap constraints: dist(i,j) >= 2r => dist^2 >= 4r^2.
    Returns array of inequalities g(x) >= 0.
    """
    r = x[-1]
    c = x[:-1].reshape(n, 2)
    res = np.empty((n * (n - 1)) // 2)
    idx = 0
    for i in range(n):
        ci = c[i]
        for j in range(i + 1, n):
            dx = ci[0] - c[j][0]
            dy = ci[1] - c[j][1]
            res[idx] = dx*dx + dy*dy - 4*r*r
            idx += 1
    return res

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = N_CIRCLES
    centers_init, r_init = generate_initial_centers(n)
    
    # Variables: [x1, y1, ..., xn, yn, r]
    x0 = np.concatenate([centers_init.flatten(), [r_init]])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)]
    
    # Constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_bounds},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # Optimization
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})
        final_x = res.x
    except Exception:
        final_x = x0

    final_r = final_x[-1]
    final_centers = final_x[:-1].reshape(n, 2)
    final_radii = np.full(n, final_r)
    
    sum_radii = float(np.sum(final_radii))
    
    return final_centers, final_radii, sum_radii
