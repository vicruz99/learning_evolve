# sol_000280 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9d8cea89) state=bb4a9139 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize


def neg_sum_r(params, n):
    """Objective: minimize negative sum of radii"""
    return -np.sum(params[2 * n:])


def all_constraints(params, n):
    """Return all inequality constraint values (must be >= 0)"""
    c = params[:2 * n].reshape(n, 2)
    r = params[2 * n:]
    out = []

    # Boundary constraints for each circle
    for i in range(n):
        out.append(c[i, 0] - r[i])
        out.append(1.0 - c[i, 0] - r[i])
        out.append(c[i, 1] - r[i])
        out.append(1.0 - c[i, 1] - r[i])

    # Non-negative radii
    for i in range(n):
        out.append(r[i])

    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            out.append(d - r[i] - r[j])

    return np.array(out)


def hex_init(n):
    """Create hexagonal grid initialization for n circles"""
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.04

    r_est = 0.085
    margin = r_est
    dy = r_est * np.sqrt(3)
    dx = 2 * r_est

    idx = 0
    for row in range(10):
        y = margin + row * dy
        if y > 1 - margin:
            break
        x0 = margin
        if row % 2 == 1:
            x0 += dx / 2
        col = 0
        while True:
            x = x0 + col * dx
            if x > 1 - margin:
                break
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
            col += 1
        if idx >= n:
            break

    # Fill remaining if any
    while idx < n:
        x = 0.5 + (idx - n + 1) * 0.01
        y = 0.5
        centers[idx] = [x, y]
        idx += 1

    return centers, radii


def adjust_validity(centers, radii, n):
    """Ensure all constraints are satisfied by iteratively fixing violations"""
    # Ensure radii respect boundaries
    for i in range(n):
        max_r = min(
            centers[i, 0],
            1 - centers[i, 0],
            centers[i, 1],
            1 - centers[i, 1]
        )
        radii[i] = min(radii[i], max(max_r, 0))

    # Fix overlaps iteratively
    for _ in range(500):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < radii[i] + radii[j] - 1e-14:
                    excess = (radii[i] + radii[j] - d + 1e-14) / 2
                    radii[i] = max(0, radii[i] - excess * 1.001)
                    radii[j] = max(0, radii[j] - excess * 1.001)
                    changed = True

        # Re-check boundaries after radius reduction
        for i in range(n):
            for dim in range(2):
                if centers[i, dim] - radii[i] < -1e-14:
                    radii[i] = max(0, centers[i, dim] + 1e-14)
                if centers[i, dim] + radii[i] > 1 + 1e-14:
                    radii[i] = max(0, 1 - centers[i, dim] - 1e-14)

        if not changed:
            break

    return centers, radii


def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Try multiple optimizations with different initializations
    for trial in range(5):
        centers, radii = hex_init(n)

        # Add random perturbation for diversity
        if trial > 0:
            rng = np.random.default_rng(1000 + trial * 137)
            centers += rng.random((n, 2)) * 0.04
            centers = np.clip(centers, 0.02, 0.98)

        params0 = np.concatenate([centers.flatten(), radii])

        # Bounds: centers in [0,1], radii in [1e-10, 0.5]
        bounds = []
        for _ in range(2 * n):
            bounds.append((0, 1))
        for _ in range(n):
            bounds.append((1e-10, 0.5))

        constraint = {
            'type': 'ineq',
            'fun': all_constraints,
            'args': (n,)
        }

        try:
            result = minimize(
                neg_sum_r,
                params0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraint,
                options={
                    'maxiter': 8000,
                    'ftol': 1e-16,
                    'disp': False
                }
            )

            co = result.x[:2 * n].reshape(n, 2)
            ro = result.x[2 * n:]

            # Ensure validity
            co, ro = adjust_validity(co, ro, n)

            current_sum = np.sum(ro)

            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = co.copy()
                best_radii = ro.copy()

        except Exception:
            continue

    # Fallback if all optimizations failed
    if best_centers is None:
        best_centers, best_radii = hex_init(n)
        best_sum = float(np.sum(best_radii))

    return best_centers, best_radii, float(best_sum)
