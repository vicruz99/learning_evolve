import numpy as np
from scipy.optimize import minimize


def compute_overlap_penalty(centers, radii):
    """Compute total overlap penalty squared."""
    n = centers.shape[0]
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += overlap ** 2
    return penalty


def compute_boundary_penalty(centers, radii):
    """Compute penalty for circles outside boundary."""
    n = centers.shape[0]
    penalty = 0.0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x < r:
            penalty += (r - x) ** 2
        if x > 1 - r:
            penalty += (x - (1 - r)) ** 2
        if y < r:
            penalty += (r - y) ** 2
        if y > 1 - r:
            penalty += (y - (1 - r)) ** 2
    return penalty


def initialize_hexagonal(n):
    """Initialize circles in hexagonal pattern."""
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.09

    row_configs = [
        (6, 0.10, 0.0),
        (5, 0.10 + 0.09 * np.sqrt(3), 0.09),
        (6, 0.10 + 2 * 0.09 * np.sqrt(3), 0.0),
        (5, 0.10 + 3 * 0.09 * np.sqrt(3), 0.09),
        (4, 0.10 + 4 * 0.09 * np.sqrt(3), 0.0),
    ]

    idx = 0
    for row, (count, y_base, offset) in enumerate(row_configs):
        available_width = 1 - 2 * 0.08
        col_spacing = available_width / (count + 1)
        for col in range(count):
            if idx >= n:
                break
            x = 0.08 + (col + 1) * col_spacing + offset * 0.5
            centers[idx] = [np.clip(x, 0.08, 0.92), np.clip(y_base, 0.08, 0.92)]
            idx += 1

    return centers, radii


def optimize_packing(centers, radii, seed_val):
    """Optimize a packing using repulsion-based method."""
    n = centers.shape[0]
    rng = np.random.RandomState(seed_val)

    for iteration in range(4000):
        # Gradually increase radii
        growth = 0.00004 * max(0.05, 1 - iteration / 4000)
        for i in range(n):
            radii[i] += growth

        # Compute repulsion forces
        forces = np.zeros((n, 2))

        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist_sq = np.sum(diff ** 2)
                dist = np.sqrt(dist_sq) + 1e-12
                min_dist = radii[i] + radii[j]

                if dist < min_dist:
                    overlap = min_dist - dist
                    strength = overlap / dist * 300
                    forces[i] += diff * strength
                    forces[j] -= diff * strength
                elif dist < min_dist + 0.015:
                    proximity = (min_dist + 0.015 - dist)
                    strength = proximity / dist * 20
                    forces[i] += diff * strength
                    forces[j] -= diff * strength

            # Boundary repulsion
            for dim in range(2):
                if centers[i, dim] < radii[i]:
                    forces[i, dim] += (radii[i] - centers[i, dim]) * 500
                if centers[i, dim] > 1 - radii[i]:
                    forces[i, dim] -= (centers[i, dim] - (1 - radii[i])) * 500

        # Apply forces with adaptive step
        step_size = 0.03 * max(0.01, 1 - iteration / 4000)
        centers += forces * step_size

        # Random perturbation for escaping local minima
        noise_level = 0.003 * max(0, 1 - iteration / 1500)
        centers += rng.randn(n, 2) * noise_level

        # Clamp to valid range
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])

    return centers, radii


def cleanup_and_validate(centers, radii, n):
    """Ensure packing is valid by resolving any remaining violations."""
    for _ in range(200):
        max_violation = 0

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                violation = radii[i] + radii[j] - dist
                if violation > max_violation:
                    max_violation = violation

            for dim in range(2):
                if centers[i, dim] < radii[i]:
                    violation = radii[i] - centers[i, dim]
                    max_violation = max(max_violation, violation)
                if centers[i, dim] > 1 - radii[i]:
                    violation = centers[i, dim] - (1 - radii[i])
                    max_violation = max(max_violation, violation)

        if max_violation < 1e-10:
            break

        # Reduce all radii slightly to resolve violations
        for i in range(n):
            radii[i] *= 0.999

    # Final boundary enforcement
    for i in range(n):
        r = max(0, radii[i])
        x, y = centers[i]
        centers[i, 0] = np.clip(x, r, 1 - r)
        centers[i, 1] = np.clip(y, r, 1 - r)
        radii[i] = min(radii[i], x - r + 1e-12, 1 - x - r + 1e-12, y - r + 1e-12, 1 - y - r + 1e-12, r)

    return centers, radii


def run_packing():
    n = 26

    best_sum = 0.0
    best_centers = None
    best_radii = None

    # Try multiple initializations
    for trial in range(20):
        if trial < 10:
            # Hexagonal initializations
            centers, radii = initialize_hexagonal(n)
            # Add small random perturbation
            rng = np.random.RandomState(trial)
            centers += rng.randn(n, 2) * 0.02
            centers[:, 0] = np.clip(centers[:, 0], 0.05, 0.95)
            centers[:, 1] = np.clip(centers[:, 1], 0.05, 0.95)
        else:
            # Random initializations with grid structure
            centers = np.zeros((n, 2))
            radii = np.ones(n) * 0.09
            rng = np.random.RandomState(trial)
            idx = 0
            for row in range(6):
                y = rng.uniform(0.1, 0.9) + row * 0.13
                y = np.clip(y, 0.1, 0.9)
                cols = 5 if row % 2 == 0 else 5
                for col in range(cols):
                    if idx >= n:
                        break
                    x = rng.uniform(0.1, 0.9) + col * 0.15
                    x = np.clip(x, 0.1, 0.9)
                    centers[idx] = [x, y]
                    idx += 1

        # Optimize
        opt_centers, opt_radii = optimize_packing(centers, radii, trial * 137)

        # Cleanup
        opt_centers, opt_radii = cleanup_and_validate(opt_centers, opt_radii, n)

        # Verify validity
        valid = True
        for i in range(n):
            if opt_radii[i] < 0:
                valid = False
                break
            x, y = opt_centers[i]
            r = opt_radii[i]
            if x < r - 1e-12 or x > 1 - r + 1e-12 or y < r - 1e-12 or y > 1 - r + 1e-12:
                valid = False
                break
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((opt_centers[i] - opt_centers[j]) ** 2))
                if dist < opt_radii[i] + opt_radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            s = np.sum(opt_radii)
            if s > best_sum:
                best_sum = s
                best_centers = opt_centers.copy()
                best_radii = opt_radii.copy()

    # If still invalid, do final aggressive cleanup
    if best_centers is None:
        centers, radii = initialize_hexagonal(n)
        centers, radii = optimize_packing(centers, radii, 999)
        centers, radii = cleanup_and_validate(centers, radii, n)
        best_centers = centers
        best_radii = radii
        best_sum = np.sum(radii)

    return best_centers, best_radii, best_sum