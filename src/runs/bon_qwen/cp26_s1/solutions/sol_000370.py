# sol_000370 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b90f636d) state=d4e3808f sum of radii=2.571561 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize


N = 26


def objective(x):
    """Objective: maximize sum of radii (minimize negative)."""
    return -np.sum(x[2 * N:])


def constraint_func(x):
    """Boundary and non-overlap constraints (all >= 0)."""
    c = []
    ci = x[:2 * N].reshape(N, 2)
    ri = x[2 * N:]

    # Boundary constraints
    for i in range(N):
        c.append(ci[i, 0] - ri[i])
        c.append(1 - ci[i, 0] - ri[i])
        c.append(ci[i, 1] - ri[i])
        c.append(1 - ci[i, 1] - ri[i])

    # Non-overlap constraints using squared distance
    for i in range(N):
        for j in range(i + 1, N):
            dx = ci[i, 0] - ci[j, 0]
            dy = ci[i, 1] - ci[j, 1]
            min_d = ri[i] + ri[j]
            c.append(dx * dx + dy * dy - min_d * min_d)

    return np.array(c)


def hexagonal_init(n, n_rows, n_cols):
    """Initialize circles in hexagonal lattice pattern."""
    r_width = 1.0 / (2.0 * n_cols)
    r_height = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3))
    r = min(r_width, r_height) * 0.93

    centers = np.zeros((n, 2))
    radii = np.full(n, r)

    idx = 0
    for row in range(n_rows):
        for col in range(n_cols):
            if idx >= n:
                break
            x = col * 2.0 * r + r
            y = row * r * np.sqrt(3) + r
            centers[idx] = [x, y]
            idx += 1
        if idx >= n:
            break

    return centers, radii


def random_init(n, seed):
    """Random initialization with circles packed roughly uniformly."""
    np.random.seed(seed)
    centers = np.random.rand(n, 2) * 0.8 + 0.1
    radii = np.full(n, 0.05)
    return centers, radii


def force_refine(centers, radii, iterations=300, lr=0.005):
    """Force-based refinement to resolve tight constraints."""
    n = len(radii)
    centers = centers.copy().astype(np.float64)
    radii = radii.copy().astype(np.float64)

    for it in range(iterations):
        forces = np.zeros_like(centers)
        current_lr = lr * (1.0 - 0.5 * it / iterations)

        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist_sq = np.sum(diff ** 2)
                dist = np.sqrt(dist_sq + 1e-16)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    push = (min_dist - dist) / (dist + 1e-10) * current_lr
                    forces[i] += diff * push
                    forces[j] -= diff * push

        for i in range(n):
            for dim in range(2):
                if centers[i, dim] < radii[i]:
                    forces[i, dim] += current_lr * (radii[i] - centers[i, dim])
                if centers[i, dim] > 1 - radii[i]:
                    forces[i, dim] -= current_lr * (centers[i, dim] - (1 - radii[i]))

        centers += forces
        centers = np.clip(centers, 1e-8, 1 - 1e-8)

    return centers, radii


def run_packing():
    n = N
    best_centers = None
    best_radii = None
    best_sum = 0.0

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}

    # Try multiple hexagonal grid configurations
    grid_configs = [
        (5, 5), (5, 6), (6, 5), (4, 7), (7, 4),
        (6, 4), (4, 6), (6, 6), (5, 7), (7, 5),
        (4, 8), (8, 4), (3, 9), (9, 3), (5, 8), (8, 5)
    ]

    for n_rows, n_cols in grid_configs:
        centers, radii = hexagonal_init(n, n_rows, n_cols)
        x0 = np.concatenate([centers.flatten(), radii])

        result = minimize(
            objective, x0, bounds=bounds, constraints=cons,
            method='SLSQP',
            options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False}
        )

        cur_sum = -result.fun
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = result.x[:2 * n].reshape(n, 2).copy()
            best_radii = result.x[2 * n:].copy()

    # Try random initializations
    for seed in range(10):
        centers, radii = random_init(n, seed)
        x0 = np.concatenate([centers.flatten(), radii])

        result = minimize(
            objective, x0, bounds=bounds, constraints=cons,
            method='SLSQP',
            options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False}
        )

        cur_sum = -result.fun
        if cur_sum > best_sum:
            best_sum = cur_sum
            best_centers = result.x[:2 * n].reshape(n, 2).copy()
            best_radii = result.x[2 * n:].copy()

    # Force-based refinement
    best_centers, best_radii = force_refine(best_centers, best_radii, iterations=500, lr=0.003)

    # Re-optimize after refinement
    x0 = np.concatenate([best_centers.flatten(), best_radii])

    result = minimize(
        objective, x0, bounds=bounds, constraints=cons,
        method='SLSQP',
        options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False}
    )

    best_centers = result.x[:2 * n].reshape(n, 2)
    best_radii = result.x[2 * n:]
    best_sum = -result.fun

    # Second round of force refinement and optimization
    best_centers, best_radii = force_refine(best_centers, best_radii, iterations=500, lr=0.002)

    x0 = np.concatenate([best_centers.flatten(), best_radii])

    result = minimize(
        objective, x0, bounds=bounds, constraints=cons,
        method='SLSQP',
        options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False}
    )

    best_centers = result.x[:2 * n].reshape(n, 2)
    best_radii = result.x[2 * n:]
    best_sum = -result.fun

    return best_centers, best_radii, best_sum
