# sol_000006 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5f9158d7) state=62f34940 sum of radii=2.607803 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math


def run_packing():
    n = 26
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    best_sum = 0.0

    for trial in range(8):
        centers, radii = create_initial_packing(n, trial)
        centers, radii = optimize_packing(centers, radii, n)
        s = np.sum(radii)
        if s > best_sum:
            best_sum = s
            best_centers = centers.copy()
            best_radii = radii.copy()

    return best_centers, best_radii, float(best_sum)


def create_initial_packing(n, seed):
    rng = np.random.RandomState(seed)
    centers = np.zeros((n, 2))

    # Hexagonal arrangement: 6-5-6-5-4 = 26 circles
    row_configs = [(6, 0.0), (5, 0.5), (6, 0.0), (5, 0.5), (4, 0.0)]

    idx = 0
    num_rows = len(row_configs)
    for row_idx, (count, offset) in enumerate(row_configs):
        y = (row_idx + 1.0) / (num_rows + 1.0)
        for col in range(count):
            x = (col + 1.0) / (count + 1.0) + offset / (count + 1.0)
            x = np.clip(x + rng.uniform(-0.02, 0.02), 0.01, 0.99)
            y = np.clip(y + rng.uniform(-0.02, 0.02), 0.01, 0.99)
            centers[idx] = [x, y]
            idx += 1

    radii = compute_initial_radii(centers, n)
    return centers, radii


def compute_initial_radii(centers, n):
    radii = np.zeros(n)
    for i in range(n):
        r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if i != j:
                d = math.sqrt((centers[i, 0] - centers[j, 0]) ** 2 + (centers[i, 1] - centers[j, 1]) ** 2)
                r = min(r, d / 2.0 * 0.92)
        radii[i] = max(0.001, r)
    return radii


def constraint_r_leq_x(x, i, n):
    return x[2 * i] - x[2 * n + i]


def constraint_r_leq_1x(x, i, n):
    return 1.0 - x[2 * i] - x[2 * n + i]


def constraint_r_leq_y(x, i, n):
    return x[2 * i + 1] - x[2 * n + i]


def constraint_r_leq_1y(x, i, n):
    return 1.0 - x[2 * i + 1] - x[2 * n + i]


def constraint_r_pos(x, i, n):
    return x[2 * n + i]


def constraint_no_overlap(x, i, j, n):
    xi = x[2 * i]
    yi = x[2 * i + 1]
    xj = x[2 * j]
    yj = x[2 * j + 1]
    ri = x[2 * n + i]
    rj = x[2 * n + j]
    dx = xi - xj
    dy = yi - yj
    dist = math.sqrt(dx * dx + dy * dy)
    return dist - ri - rj


def objective_func(x, n):
    return -np.sum(x[2 * n:])


def optimize_packing(centers, radii, n):
    x0 = np.concatenate([centers.flatten(), radii])

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': constraint_r_leq_x, 'args': (i, n)})
        cons.append({'type': 'ineq', 'fun': constraint_r_leq_1x, 'args': (i, n)})
        cons.append({'type': 'ineq', 'fun': constraint_r_leq_y, 'args': (i, n)})
        cons.append({'type': 'ineq', 'fun': constraint_r_leq_1y, 'args': (i, n)})
        cons.append({'type': 'ineq', 'fun': constraint_r_pos, 'args': (i, n)})

    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': constraint_no_overlap, 'args': (i, j, n)})

    result = minimize(objective_func, x0, method='SLSQP', args=(n,),
                      bounds=bounds, constraints=cons,
                      options={'maxiter': 15000, 'ftol': 1e-15, 'disp': False})

    opt_centers = result.x[:2 * n].reshape((n, 2))
    opt_radii = result.x[2 * n:]

    # Ensure radii are non-negative and within bounds
    opt_radii = np.maximum(opt_radii, 0.0)
    for i in range(n):
        r_max = min(opt_centers[i, 0], 1.0 - opt_centers[i, 0],
                     opt_centers[i, 1], 1.0 - opt_centers[i, 1])
        opt_radii[i] = min(opt_radii[i], r_max)

    # Iteratively fix any remaining overlaps by reducing radii
    for _ in range(300):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = opt_centers[i, 0] - opt_centers[j, 0]
                dy = opt_centers[i, 1] - opt_centers[j, 1]
                d = math.sqrt(dx * dx + dy * dy)
                if d < opt_radii[i] + opt_radii[j] - 1e-10:
                    overlap = opt_radii[i] + opt_radii[j] - d
                    opt_radii[i] -= overlap / 2.0
                    opt_radii[j] -= overlap / 2.0
                    changed = True
        if not changed:
            break

    return opt_centers, opt_radii
