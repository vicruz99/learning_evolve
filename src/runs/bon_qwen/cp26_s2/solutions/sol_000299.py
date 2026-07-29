# sol_000299 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d060b5cc) state=ba80cb01 sum of radii=2.603408 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(v):
    """Maximize sum of radii (minimize negative sum)."""
    return -np.sum(v[2*N_CIRCLES:])

def constraint_func(v):
    """Returns vector of inequality constraints >= 0."""
    n = N_CIRCLES
    centers = v[:2*n].reshape(n, 2)
    radii = v[2*n:]

    # Pairwise distance constraints: dist(i,j) - r_i - r_j >= 0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Upper triangular mask for unique pairs
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pair_cons = (dists[mask] - r_sum[mask]).flatten()

    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    b_cons = np.hstack([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])

    # Non-negative radii
    return np.concatenate([pair_cons, b_cons, radii])

def get_initial_guess():
    """Generate a hexagonal grid initial configuration."""
    pts = []
    sqrt3 = np.sqrt(3)
    rows_cfg = [5, 6, 5, 6, 4]
    y_pos = 0.0
    for cnt in rows_cfg:
        offset = 0.0 if len(pts) == 0 else 1.0
        for k in range(cnt):
            pts.append([2*k + offset, y_pos])
        y_pos += sqrt3
        if len(pts) >= N_CIRCLES:
            break
    pts = pts[:N_CIRCLES]

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Scale to leave margin for expansion
    sx = 0.8 / (max_x - min_x)
    sy = 0.8 / (max_y - min_y)
    scale = min(sx, sy)

    centers = np.array(pts) * scale
    centers[:, 0] += 0.5 - scale * (min_x + max_x) / 2
    centers[:, 1] += 0.5 - scale * (min_y + max_y) / 2

    radii = np.full(N_CIRCLES, 0.04)
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    n = N_CIRCLES
    v0 = get_initial_guess()
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}

    res = minimize(objective_func, v0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})

    centers = res.x[:2*n].reshape(n, 2)
    radii = res.x[2*n:]

    # Clamp to valid ranges
    centers = np.clip(centers, 0.0, 1.0)
    radii = np.clip(radii, 0.0, 0.5)

    # Iterative shrink to strictly satisfy validation tolerance if numerical drift occurs
    for _ in range(3):
        max_violation = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                req = radii[i] + radii[j]
                if dist < req - 1e-12:
                    factor = dist / req
                    radii[i] *= factor
                    radii[j] *= factor
            for coord in (0, 1):
                if centers[i, coord] - radii[i] < -1e-12:
                    radii[i] = centers[i, coord]
                if centers[i, coord] + radii[i] > 1.0 + 1e-12:
                    radii[i] = 1.0 - centers[i, coord]

    return centers, radii, float(np.sum(radii))
