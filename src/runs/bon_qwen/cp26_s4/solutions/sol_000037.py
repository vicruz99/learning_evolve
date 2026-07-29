# sol_000037 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e09efbf) state=1ca85373 sum of radii=2.490492 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26


def overlap_constraint(state, i, j, n):
    ci_x = state[2 * i]
    ci_y = state[2 * i + 1]
    cj_x = state[2 * j]
    cj_y = state[2 * j + 1]
    ri = state[2 * n + i]
    rj = state[2 * n + j]
    dx = ci_x - cj_x
    dy = ci_y - cj_y
    dist = np.sqrt(dx * dx + dy * dy)
    return dist - ri - rj


def boundary_left_constraint(state, i, n):
    return state[2 * i] - state[2 * n + i]


def boundary_right_constraint(state, i, n):
    return 1.0 - state[2 * i] - state[2 * n + i]


def boundary_bottom_constraint(state, i, n):
    return state[2 * i + 1] - state[2 * n + i]


def boundary_top_constraint(state, i, n):
    return 1.0 - state[2 * i + 1] - state[2 * n + i]


def radius_positive_constraint(state, i, n):
    return state[2 * n + i]


def objective_func(state):
    n = N_CIRCLES
    return -np.sum(state[2 * n:])


def create_hexagonal_initial():
    n = N_CIRCLES
    centers = []
    r_init = 0.06

    for row in range(8):
        y = r_init + row * r_init * np.sqrt(3)
        if y + r_init > 1.0:
            break
        if row % 2 == 0:
            for col in range(7):
                x = r_init + col * 2 * r_init
                if x + r_init <= 1.0:
                    centers.append([x, y])
        else:
            for col in range(6):
                x = 2 * r_init + col * 2 * r_init
                if x + r_init <= 1.0:
                    centers.append([x, y])
        if len(centers) >= n:
            break

    centers = np.array(centers[:n])
    radii = np.ones(n) * 0.025
    return np.concatenate([centers.flatten(), radii])


def force_based_simulation(centers, radii, n, iterations=3000):
    centers = centers.copy()
    radii = radii.copy()

    for step in range(iterations):
        radii += 5e-6

        forces = np.zeros_like(centers)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist_sq = np.sum(diff * diff)
                dist = np.sqrt(dist_sq) if dist_sq > 1e-15 else 1e-15
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    repulsion = (min_dist - dist) / (dist + 1e-15)
                    forces[i] += diff * repulsion
                    forces[j] -= diff * repulsion

        # Boundary repulsion
        for i in range(n):
            r = radii[i]
            if centers[i, 0] < r:
                forces[i, 0] += (r - centers[i, 0]) * 10
            if centers[i, 0] > 1 - r:
                forces[i, 0] -= (centers[i, 0] - (1 - r)) * 10
            if centers[i, 1] < r:
                forces[i, 1] += (r - centers[i, 1]) * 10
            if centers[i, 1] > 1 - r:
                forces[i, 1] -= (centers[i, 1] - (1 - r)) * 10

        step_size = 0.005 / (1 + step * 0.0005)
        centers += forces * step_size
        centers = np.clip(centers, 0, 1)

    return centers, radii


def fix_overlaps(centers, radii, n):
    centers = centers.copy()
    radii = radii.copy()

    for _ in range(50):
        max_violation = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j]:
                    violation = radii[i] + radii[j] - dist
                    if violation > max_violation:
                        max_violation = violation
                    scale = dist / (radii[i] + radii[j]) * 0.999
                    radii[i] *= scale
                    radii[j] *= scale

        # Boundary fixing
        for i in range(n):
            r = radii[i]
            if centers[i, 0] < r:
                radii[i] = centers[i, 0] * 0.999
            if centers[i, 0] > 1 - r:
                radii[i] = (1 - centers[i, 0]) * 0.999
            if centers[i, 1] < r:
                radii[i] = centers[i, 1] * 0.999
            if centers[i, 1] > 1 - r:
                radii[i] = (1 - centers[i, 1]) * 0.999

        radii = np.maximum(radii, 1e-10)

    return centers, radii


def run_packing():
    n = N_CIRCLES

    # Phase 1: Create initial hexagonal grid
    x0 = create_hexagonal_initial()
    centers_init = x0[:2 * n].reshape((n, 2))
    radii_init = x0[2 * n:]

    # Phase 2: Force-based simulation to expand and refine
    centers_sim, radii_sim = force_based_simulation(centers_init, radii_init, n, iterations=3000)

    # Phase 3: Constrained optimization starting from simulation result
    x0_opt = np.concatenate([centers_sim.flatten(), radii_sim])

    constraints = []

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': overlap_constraint, 'args': (i, j, n)})

    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': boundary_left_constraint, 'args': (i, n)})
        constraints.append({'type': 'ineq', 'fun': boundary_right_constraint, 'args': (i, n)})
        constraints.append({'type': 'ineq', 'fun': boundary_bottom_constraint, 'args': (i, n)})
        constraints.append({'type': 'ineq', 'fun': boundary_top_constraint, 'args': (i, n)})
        constraints.append({'type': 'ineq', 'fun': radius_positive_constraint, 'args': (i, n)})

    bounds = [(0, 1) for _ in range(2 * n)] + [(1e-8, 0.5) for _ in range(n)]

    # Run optimization
    result = minimize(
        objective_func,
        x0_opt,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False}
    )

    centers_opt = result.x[:2 * n].reshape((n, 2))
    radii_opt = result.x[2 * n:]

    # Phase 4: Fix any remaining violations
    centers_opt, radii_opt = fix_overlaps(centers_opt, radii_opt, n)

    centers_opt = np.clip(centers_opt, 0, 1)
    radii_opt = np.maximum(radii_opt, 0)

    sum_radii = np.sum(radii_opt)

    return centers_opt, radii_opt, sum_radii
