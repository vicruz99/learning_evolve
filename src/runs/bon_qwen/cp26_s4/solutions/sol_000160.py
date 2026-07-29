# sol_000160 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2296af5d) state=93264094 sum of radii=2.626247 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize


def objective(params, n):
    return -np.sum(params[2 * n :])


def all_constraints(params, n):
    constraints = []
    for i in range(n):
        x = params[2 * i]
        y = params[2 * i + 1]
        r = params[2 * n + i]
        constraints.append(x - r)
        constraints.append(1.0 - x - r)
        constraints.append(y - r)
        constraints.append(1.0 - y - r)
        constraints.append(r)

    for i in range(n):
        for j in range(i + 1, n):
            dx = params[2 * i] - params[2 * j]
            dy = params[2 * i + 1] - params[2 * j + 1]
            dist = np.sqrt(dx * dx + dy * dy)
            constraints.append(dist - params[2 * n + i] - params[2 * n + j])

    return np.array(constraints)


def make_initial_config(n, pattern, seed=42):
    centers = []

    if pattern == "hex454":
        row_counts = [4, 5, 4, 5, 4, 4]
    elif pattern == "hex565":
        row_counts = [5, 6, 5, 6, 4]
    elif pattern == "hex545":
        row_counts = [5, 4, 5, 4, 5, 3]
    elif pattern == "hex354":
        row_counts = [3, 5, 4, 5, 4, 5]
    elif pattern == "hex555":
        row_counts = [5, 5, 5, 5, 5, 1]
    else:
        np.random.seed(seed)
        for _ in range(n):
            centers.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
        centers = np.array(centers)
        radii = np.full(n, 0.07)
        return np.concatenate([centers.flatten(), radii])

    for row_idx, count in enumerate(row_counts):
        y = (row_idx + 0.5) * (1.0 / len(row_counts)) + 0.015
        if row_idx % 2 == 0:
            x_start = 0.03
        else:
            x_start = 0.12
        for col in range(count):
            if count > 1:
                x = x_start + col * (0.94 / (count - 1))
            else:
                x = 0.5
            centers.append([x, y])

    centers = np.array(centers[:n])
    radii = np.full(n, 0.075)
    return np.concatenate([centers.flatten(), radii])


def solve_from_init(x0, n):
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    constraints = [{"type": "ineq", "fun": all_constraints, "args": (n,)}]

    result = minimize(
        objective,
        x0,
        args=(n,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 5000, "ftol": 1e-15, "disp": False},
    )

    return result


def enforce_feasibility(centers, radii, n, margin=1e-10):
    centers = np.clip(centers, margin, 1.0 - margin)
    radii = np.maximum(radii, margin)

    for _ in range(50):
        changed = False
        for i in range(n):
            max_r = min(
                centers[i, 0] - margin,
                1.0 - centers[i, 0] - margin,
                centers[i, 1] - margin,
                1.0 - centers[i, 1] - margin,
            )
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    max_r = min(max_r, dist - radii[j] - margin)
            if radii[i] > max_r:
                radii[i] = max(0.0001, max_r)
                changed = True
        if not changed:
            break

    return centers, radii


def run_packing():
    n = 26

    patterns = ["hex454", "hex565", "hex545", "hex354", "hex555", "random"]

    best_sum = 0.0
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)

    for pattern in patterns:
        x0 = make_initial_config(n, pattern, seed=42)

        result = solve_from_init(x0, n)

        current_sum = -result.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = result.x[: 2 * n].reshape(n, 2)
            best_radii = np.maximum(result.x[2 * n :], 0)

    # Refinement: use the best solution as starting point for another round
    x0_refine = np.concatenate([best_centers.flatten(), best_radii])

    # Perturb slightly and re-optimize
    np.random.seed(123)
    for trial in range(3):
        perturbed = x0_refine.copy()
        perturbed[: 2 * n] += np.random.uniform(-0.005, 0.005, 2 * n)
        perturbed[: 2 * n] = np.clip(perturbed[: 2 * n], 0.01, 0.99)
        perturbed[2 * n :] = np.maximum(perturbed[2 * n :], 0.001)

        result = solve_from_init(perturbed, n)
        current_sum = -result.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = result.x[: 2 * n].reshape(n, 2)
            best_radii = np.maximum(result.x[2 * n :], 0)

    # Enforce feasibility
    best_centers, best_radii = enforce_feasibility(best_centers, best_radii, n)

    final_sum = np.sum(best_radii)

    return best_centers, best_radii, final_sum
