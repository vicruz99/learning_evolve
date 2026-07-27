import numpy as np
from scipy.optimize import minimize


def run_packing():
    n = 26

    best_sum = 0.0
    best_centers = None
    best_radii = None

    for seed in range(80):
        centers, radii, s = try_packing(n, seed)

        if s > best_sum:
            best_sum = s
            best_centers = centers.copy()
            best_radii = radii.copy()

    return best_centers, best_radii, best_sum


def try_packing(n, seed):
    np.random.seed(seed)

    if seed < 30:
        centers = init_hexagonal(n, seed)
    elif seed < 50:
        centers = init_grid(n, seed)
    else:
        centers = init_random(n, seed)

    radii = np.ones(n) * 0.015

    centers, radii = physics_grow(centers, radii, n)

    centers, radii = optimize_final(centers, radii, n)

    return centers, radii, np.sum(radii)


def init_hexagonal(n, seed):
    centers = np.zeros((n, 2))
    idx = 0

    configs = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [5, 5, 5, 5, 6],
        [6, 6, 5, 5, 4],
        [4, 6, 5, 6, 5],
        [6, 5, 5, 6, 4],
        [5, 5, 6, 5, 5],
        [6, 5, 4, 6, 5],
        [5, 4, 6, 5, 6],
        [6, 4, 5, 6, 5],
        [5, 6, 4, 5, 6],
        [4, 5, 6, 5, 6],
        [6, 6, 6, 4, 4],
        [6, 6, 4, 6, 4],
        [6, 4, 6, 6, 4],
        [4, 6, 6, 6, 4],
        [5, 5, 5, 6, 5],
        [5, 5, 6, 5, 5],
        [5, 6, 5, 5, 5],
        [6, 5, 5, 5, 5],
        [5, 5, 5, 5, 6],
        [6, 5, 5, 5, 5],
        [5, 6, 5, 5, 5],
        [5, 5, 6, 5, 5],
        [5, 5, 5, 6, 5],
        [4, 6, 6, 5, 5],
        [5, 5, 6, 6, 4],
        [6, 4, 6, 5, 5],
        [5, 5, 4, 6, 6],
        [5, 4, 5, 6, 6],
    ]

    config = configs[seed % len(configs)]

    np.random.seed(seed * 7 + 3)
    perturb = 0.015

    for row, count in enumerate(config):
        for col in range(count):
            if idx < n:
                x = (col + 0.5) / count
                if row % 2 == 1:
                    x += 1.0 / (2 * count)
                y = (row + 0.5) / len(config)

                x += np.random.randn() * perturb
                y += np.random.randn() * perturb

                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)

                centers[idx] = [x, y]
                idx += 1

    return centers


def init_grid(n, seed):
    centers = np.zeros((n, 2))
    idx = 0

    np.random.seed(seed * 13 + 7)
    perturb = 0.02

    grid_sizes = [
        (5, 5, 1),
        (6, 5, 0),
        (5, 6, 0),
    ]
    gs = grid_sizes[seed % len(grid_sizes)]
    rows, cols, extra = gs

    for row in range(rows):
        for col in range(cols):
            if idx < n:
                x = (col + 0.5) / cols + np.random.randn() * perturb
                y = (row + 0.5) / rows + np.random.randn() * perturb
                centers[idx] = [np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)]
                idx += 1

    if extra > 0 and idx < n:
        centers[idx] = [0.5 + np.random.randn() * 0.1, 0.5 + np.random.randn() * 0.1]
        centers[idx] = np.clip(centers[idx], 0.05, 0.95)
        idx += 1

    return centers


def init_random(n, seed):
    np.random.seed(seed * 31 + 11)
    centers = np.random.rand(n, 2) * 0.8 + 0.1
    return centers


def physics_grow(centers, radii, n):
    for grow_step in range(3000):
        dr = 5e-5
        radii += dr

        for relax_step in range(200):
            forces = np.zeros_like(centers)

            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    dist_sq = dx * dx + dy * dy
                    dist = np.sqrt(dist_sq + 1e-15)

                    overlap = radii[i] + radii[j] - dist
                    if overlap > 0:
                        scale = overlap / dist * 0.3
                        forces[i, 0] += scale * dx
                        forces[i, 1] += scale * dy
                        forces[j, 0] -= scale * dx
                        forces[j, 1] -= scale * dy

            for i in range(n):
                for d in range(2):
                    if centers[i, d] < radii[i]:
                        forces[i, d] += (radii[i] - centers[i, d]) * 2.0
                    elif centers[i, d] > 1 - radii[i]:
                        forces[i, d] -= (centers[i, d] - (1 - radii[i])) * 2.0

            centers += forces
            centers = np.clip(centers, 1e-6, 1 - 1e-6)

            if np.max(np.abs(forces)) < 1e-9:
                break

    return centers, radii


def penalized_objective(x, n, weight):
    centers = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    obj = -np.sum(radii)

    penalty = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += overlap * overlap

    for i in range(n):
        for d in range(2):
            violation = radii[i] - centers[i, d]
            if violation > 0:
                penalty += violation * violation
            violation = radii[i] - (1.0 - centers[i, d])
            if violation > 0:
                penalty += violation * violation

    return obj + weight * penalty


def optimize_final(centers, radii, n):
    x0 = np.concatenate([centers.flatten(), radii])

    bounds = []
    for i in range(n):
        bounds.extend([(0.0001, 0.9999), (0.0001, 0.9999), (0.0, 0.5)])

    best_x = x0.copy()
    weight = 100.0

    for iteration in range(400):
        weight = min(weight * 1.012, 500000.0)

        result = minimize(
            penalized_objective,
            best_x,
            args=(n, weight),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 80, "ftol": 1e-15, "gtol": 1e-10},
        )

        best_x = result.x.copy()

        if iteration > 100 and iteration % 50 == 0:
            best_x += np.random.randn(len(best_x)) * 1e-5

    centers = best_x[:2 * n].reshape(n, 2)
    radii = best_x[2 * n:]

    radii = np.maximum(radii, 0.0)

    return centers, radii