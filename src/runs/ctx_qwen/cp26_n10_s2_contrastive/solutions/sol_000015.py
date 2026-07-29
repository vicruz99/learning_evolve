# sol_000015 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3abe07e0) state=9fd6082b sum of radii=2.622453 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math


def obj_func(params, n):
    """Negative sum of radii (we minimize this)."""
    radii = params[2 * n :]
    return -np.sum(radii)


def all_constraints(params, n):
    """Return all constraint values (must be >= 0)."""
    centers = params[: 2 * n].reshape(n, 2)
    radii = params[2 * n :]

    cons = []

    # Boundary constraints: 4 per circle
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        cons.append(x - r)          # left: x >= r
        cons.append(1.0 - x - r)    # right: x + r <= 1
        cons.append(y - r)          # bottom: y >= r
        cons.append(1.0 - y - r)    # top: y + r <= 1

    # Overlap constraints: C(n,2) pairs
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            cons.append(dist - radii[i] - radii[j])

    return np.array(cons)


def make_hex_initial(n, spacing=0.17, margin=0.1):
    """Create hexagonal lattice initialization."""
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.035

    idx = 0
    row = 0
    while idx < n:
        y = margin + row * spacing * math.sqrt(3) / 2
        x_start = margin + (row % 2) * spacing / 2

        col = 0
        while x_start + col * spacing <= 1 - margin and idx < n:
            centers[idx] = [x_start + col * spacing, y]
            idx += 1
            col += 1
        row += 1

    return centers, radii


def make_grid_initial(n, grid_size=5):
    """Create grid initialization."""
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.035

    idx = 0
    for row in range(grid_size):
        for col in range(grid_size):
            if idx < n:
                x = 0.08 + col * 0.17
                y = 0.08 + row * 0.17
                centers[idx] = [x, y]
                idx += 1

    # Place remaining circles in gaps
    while idx < n:
        centers[idx] = [0.5, 0.5]
        idx += 1

    return centers, radii


def fix_violations(centers, radii):
    """Fix boundary and overlap violations by scaling radii down."""
    n = len(radii)

    # Fix boundary violations
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        max_r = min(x, 1 - x, y, 1 - y)
        if max_r < r:
            radii[i] = max(0, max_r - 1e-10)

    # Iteratively reduce radii to fix overlaps
    for iteration in range(200):
        max_violation = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    max_violation = max(max_violation, min_dist - dist)

        if max_violation < 1e-12:
            break

        radii *= 0.995

    return centers, radii


def run_single_optimization(n, centers, radii, max_iter=5000):
    """Run SLSQP optimization from given initial state."""
    params = np.concatenate([centers.ravel(), radii])

    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    cons = {"type": "ineq", "fun": all_constraints, "args": (n,)}

    result = minimize(
        obj_func,
        params,
        args=(n,),
        method="SLSQP",
        bounds=bounds,
        constraints=[cons],
        options={"maxiter": max_iter, "ftol": 1e-15, "disp": False},
    )

    opt_centers = result.x[: 2 * n].reshape(n, 2)
    opt_radii = result.x[2 * n :]

    return opt_centers, opt_radii


def run_packing():
    """Main packing function. Returns (centers, radii, sum_radii)."""
    n = 26

    best_sum = -1
    best_centers = None
    best_radii = None

    # Strategy 1: Hexagonal pattern with multiple noise perturbations
    for trial in range(6):
        np.random.seed(trial * 7 + 13)
        centers, radii = make_hex_initial(n, spacing=0.16 + trial * 0.01, margin=0.08)

        if trial > 0:
            noise = np.random.randn(n, 2) * 0.015
            centers += noise
            centers = np.clip(centers, 0.03, 0.97)

        opt_centers, opt_radii = run_single_optimization(n, centers, radii)
        opt_centers, opt_radii = fix_violations(opt_centers, opt_radii)

        current_sum = np.sum(opt_radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = opt_centers.copy()
            best_radii = opt_radii.copy()

    # Strategy 2: Grid pattern
    for trial in range(3):
        np.random.seed(trial * 31 + 7)
        centers, radii = make_grid_initial(n)

        if trial > 0:
            noise = np.random.randn(n, 2) * 0.015
            centers += noise
            centers = np.clip(centers, 0.03, 0.97)

        opt_centers, opt_radii = run_single_optimization(n, centers, radii)
        opt_centers, opt_radii = fix_violations(opt_centers, opt_radii)

        current_sum = np.sum(opt_radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = opt_centers.copy()
            best_radii = opt_radii.copy()

    # Strategy 3: Use best result as seed for further refinement
    if best_centers is not None:
        for trial in range(3):
            np.random.seed(trial * 53 + 11)
            centers = best_centers.copy()
            radii = best_radii.copy()

            # Perturb slightly
            noise = np.random.randn(n, 2) * 0.005
            centers += noise
            centers = np.clip(centers, 0.01, 0.99)
            radii_noise = np.random.randn(n) * 0.002
            radii += radii_noise
            radii = np.clip(radii, 0.001, 0.5)

            opt_centers, opt_radii = run_single_optimization(n, centers, radii, max_iter=8000)
            opt_centers, opt_radii = fix_violations(opt_centers, opt_radii)

            current_sum = np.sum(opt_radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = opt_centers.copy()
                best_radii = opt_radii.copy()

    return best_centers, best_radii, best_sum
