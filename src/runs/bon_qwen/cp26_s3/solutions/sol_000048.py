# sol_000048 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state baeb2167) state=535b0586 sum of radii=2.595286 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math
from functools import partial


def bound_constraint(x, n):
    c = x[:2 * n].reshape(n, 2)
    r = x[2 * n:]
    con = []
    for i in range(n):
        con.append(c[i, 0] - r[i])
        con.append(1 - c[i, 0] - r[i])
        con.append(c[i, 1] - r[i])
        con.append(1 - c[i, 1] - r[i])
    return np.array(con)


def overlap_constraint(x, n):
    c = x[:2 * n].reshape(n, 2)
    r = x[2 * n:]
    con = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            con.append(d - r[i] - r[j])
    return np.array(con)


def objective_func(x, n):
    return -np.sum(x[2 * n:])


def make_bounds(n):
    result = []
    for i in range(2 * n):
        result.append((0.001, 0.999))
    for i in range(n):
        result.append((0.001, 0.5))
    return result


def hex_init(n):
    centers = np.zeros((n, 2))
    col_spacing = 0.19
    row_spacing = col_spacing * math.sqrt(3) / 2
    idx = 0
    row = 0
    while idx < n:
        nc = 6 if row % 2 == 0 else 5
        for c in range(nc):
            if idx < n:
                x = c * col_spacing + (col_spacing / 2 if row % 2 else 0)
                y = row * row_spacing
                centers[idx] = [x, y]
                idx += 1
        row += 1

    x_min = centers[:, 0].min()
    x_max = centers[:, 0].max()
    y_min = centers[:, 1].min()
    y_max = centers[:, 1].max()

    margin = 0.02
    x_range = x_max - x_min + 1e-12
    y_range = y_max - y_min + 1e-12
    scale_x = (1 - 2 * margin) / x_range
    scale_y = (1 - 2 * margin) / y_range
    scale = min(scale_x, scale_y)

    centers[:, 0] = (centers[:, 0] - x_min) * scale + margin
    centers[:, 1] = (centers[:, 1] - y_min) * scale + margin

    return centers


def hex_init_varied(n, offset_factor):
    centers = np.zeros((n, 2))
    col_spacing = 0.19 + offset_factor * 0.01
    row_spacing = col_spacing * math.sqrt(3) / 2
    idx = 0
    row = 0
    while idx < n:
        nc = 6 if row % 2 == 0 else 5
        for c in range(nc):
            if idx < n:
                x = c * col_spacing + (col_spacing / 2 if row % 2 else 0)
                y = row * row_spacing
                centers[idx] = [x, y]
                idx += 1
        row += 1

    x_min = centers[:, 0].min()
    x_max = centers[:, 0].max()
    y_min = centers[:, 1].min()
    y_max = centers[:, 1].max()

    margin = 0.02
    x_range = x_max - x_min + 1e-12
    y_range = y_max - y_min + 1e-12
    scale_x = (1 - 2 * margin) / x_range
    scale_y = (1 - 2 * margin) / y_range
    scale = min(scale_x, scale_y)

    centers[:, 0] = (centers[:, 0] - x_min) * scale + margin
    centers[:, 1] = (centers[:, 1] - y_min) * scale + margin

    return centers


def random_init(n, seed):
    rng = np.random.RandomState(seed)
    centers = rng.rand(n, 2) * 0.8 + 0.1
    return centers


def is_valid_solution(centers, radii, n):
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-10 or x + r > 1 + 1e-10:
            return False
        if y - r < -1e-10 or y + r > 1 + 1e-10:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = math.sqrt(dx * dx + dy * dy)
            if d < radii[i] + radii[j] - 1e-10:
                return False

    return True


def cleanup_solution(centers, radii, n):
    centers = centers.copy()
    radii = radii.copy()

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        r = min(r, x, 1 - x, y, 1 - y)
        radii[i] = max(r, 0.001)
        centers[i, 0] = max(radii[i], min(1 - radii[i], x))
        centers[i, 1] = max(radii[i], min(1 - radii[i], y))

    for iteration in range(20):
        max_overlap = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                d = math.sqrt(dx * dx + dy * dy)
                r_sum = radii[i] + radii[j]
                if d < r_sum and d > 1e-12:
                    overlap = r_sum - d
                    nx = dx / d
                    ny = dy / d
                    centers[i, 0] -= nx * overlap / 2
                    centers[i, 1] -= ny * overlap / 2
                    centers[j, 0] += nx * overlap / 2
                    centers[j, 1] += ny * overlap / 2
                    max_overlap = max(max_overlap, overlap)
        if max_overlap < 1e-12:
            break

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        r = min(r, x, 1 - x, y, 1 - y)
        radii[i] = max(r, 0.001)
        centers[i, 0] = max(radii[i], min(1 - radii[i], x))
        centers[i, 1] = max(radii[i], min(1 - radii[i], y))

    return centers, radii


def run_packing():
    n = 26
    bounds = make_bounds(n)

    best_sum = -1.0
    best_centers = None
    best_radii = None

    initial_configs = []

    centers1 = hex_init(n)
    initial_configs.append(centers1)

    centers2 = hex_init_varied(n, -2)
    initial_configs.append(centers2)

    centers3 = hex_init_varied(n, 2)
    initial_configs.append(centers3)

    for seed in range(3):
        centers_r = random_init(n, seed)
        initial_configs.append(centers_r)

    for config_idx, centers_init in enumerate(initial_configs):
        radii_init = np.ones(n) * 0.09

        x0 = np.concatenate([centers_init.ravel(), radii_init])

        x0_pert = x0 + np.random.RandomState(config_idx + 10).randn(len(x0)) * 0.002
        x0_pert[:2 * n] = np.clip(x0_pert[:2 * n], 0.01, 0.99)
        x0_pert[2 * n:] = np.clip(x0_pert[2 * n:], 0.01, 0.3)

        bc = partial(bound_constraint, n=n)
        oc = partial(overlap_constraint, n=n)
        obj = partial(objective_func, n=n)

        cons = [
            {"type": "ineq", "fun": bc},
            {"type": "ineq", "fun": oc}
        ]

        try:
            result = minimize(
                obj, x0_pert, method="SLSQP",
                bounds=bounds, constraints=cons,
                options={"maxiter": 4000, "ftol": 1e-14, "disp": False}
            )

            if result.success or (-result.fun > best_sum):
                c = result.x[:2 * n].reshape(n, 2)
                r = result.x[2 * n:]
                s = np.sum(r)

                c_clean, r_clean = cleanup_solution(c, r, n)
                s_clean = np.sum(r_clean)

                if s_clean > best_sum and is_valid_solution(c_clean, r_clean, n):
                    best_sum = s_clean
                    best_centers = c_clean.copy()
                    best_radii = r_clean.copy()
        except Exception:
            pass

    if best_centers is None:
        safe_c = np.zeros((n, 2))
        safe_r = np.ones(n) * 0.05
        idx = 0
        for i in range(6):
            for j in range(5):
                if idx < n:
                    safe_c[idx] = [0.1 + j * 0.18, 0.1 + i * 0.15]
                    idx += 1
        best_centers = safe_c
        best_radii = safe_r
        best_sum = np.sum(safe_r)

    return best_centers, best_radii, best_sum
