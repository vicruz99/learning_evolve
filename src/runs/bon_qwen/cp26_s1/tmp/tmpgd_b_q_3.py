import numpy as np
from scipy.optimize import minimize


def compute_max_radii_from_positions(centers, max_iter=200):
    """Given fixed centers, compute maximum feasible radii using iterative method."""
    n = centers.shape[0]
    # Start with boundary-limited radii
    radii = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    radii = np.maximum(radii, 1e-10)

    for _ in range(max_iter):
        old_radii = radii.copy()
        for i in range(n):
            max_r = radii[i]
            for j in range(n):
                if i != j:
                    dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                    max_r = min(max_r, dist - old_radii[j])
            radii[i] = max(0, max_r)
        radii = np.maximum(radii, 0)
        if np.max(np.abs(radii - old_radii)) < 1e-12:
            break

    return radii


def force_optimize(centers, radii, n, steps=20000, base_lr=0.002):
    """Force-based optimization to resolve overlaps and improve packing."""
    centers = centers.copy()
    radii = radii.copy()

    for step in range(steps):
        forces = np.zeros_like(centers)

        # Repulsion between overlapping circles
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist_sq = np.sum(diff ** 2)
                dist = np.sqrt(dist_sq)
                min_dist = radii[i] + radii[j]
                if dist < min_dist and dist > 1e-12:
                    overlap = min_dist - dist
                    force_mag = overlap * 200.0
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist
                elif dist < 2 * min_dist and dist > 1e-12:
                    # Soft repulsion even for non-overlapping nearby circles
                    force_mag = (2 * min_dist - dist) * 10.0
                    forces[i] += force_mag * diff / dist
                    forces[j] -= force_mag * diff / dist

        # Boundary forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0:
                forces[i, 0] += (r - x) * 300
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 300
            if y - r < 0:
                forces[i, 1] += (r - y) * 300
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 300

        # Adaptive learning rate
        lr = base_lr * (1.0 / (1.0 + step * 0.0001))
        centers += lr * forces

        # Clamp positions to valid range
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])

    return centers


def hexagonal_init(rows_pattern, base_r):
    """Initialize centers in a hexagonal lattice pattern."""
    centers = []
    for row_idx, count in enumerate(rows_pattern):
        y = base_r + row_idx * base_r * np.sqrt(3)
        if row_idx % 2 == 1:
            offset = base_r * np.sqrt(3)
        else:
            offset = 0
        for col_idx in range(count):
            x = base_r + offset + col_idx * 2 * base_r
            centers.append([x, y])
    return np.array(centers)


def square_grid_init(n_cols, n_rows, base_r, offset_x=0, offset_y=0):
    """Initialize centers in a square grid pattern."""
    centers = []
    for row in range(n_rows):
        for col in range(n_cols):
            x = base_r + offset_x + col * 2 * base_r
            y = base_r + offset_y + row * 2 * base_r
            centers.append([x, y])
    return np.array(centers)


def gradient_optimize(centers, radii, n, max_iter=2000):
    """Gradient-based optimization using scipy L-BFGS-B with penalties."""

    def objective(params):
        c = params[:2 * n].reshape(n, 2)
        r = params[2 * n:]
        r = np.maximum(r, 0)

        obj = -np.sum(r)
        penalty = 0.0

        # Boundary penalties
        for i in range(n):
            x, y = c[i]
            ri = r[i]
            penalty += max(0, ri - x) ** 2 * 5000
            penalty += max(0, x + ri - 1) ** 2 * 5000
            penalty += max(0, ri - y) ** 2 * 5000
            penalty += max(0, y + ri - 1) ** 2 * 5000

        # Overlap penalties
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                sum_r = r[i] + r[j]
                if dist < sum_r:
                    penalty += (sum_r - dist) ** 2 * 5000

        return obj + penalty

    params = np.concatenate([centers.flatten(), radii])

    # Bounds for centers: [0,1] for each coordinate, radii: [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
    for _ in range(n):
        bounds.append((0, 0.5))

    result = minimize(
        objective, params, method='L-BFGS-B', bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-12}
    )

    opt_centers = result.x[:2 * n].reshape(n, 2)
    opt_radii = np.maximum(result.x[2 * n:], 0)

    return opt_centers, opt_radii


def validate_and_fix(centers, radii, n):
    """Validate packing and fix any small violations."""
    centers = centers.copy()
    radii = radii.copy()

    # Ensure centers are within bounds given radii
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])

    # Iteratively reduce radii to resolve overlaps
    for _ in range(100):
        max_violation = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    violation = sum_r - dist
                    max_violation = max(max_violation, violation)
                    # Reduce both radii proportionally
                    reduction = violation * 0.5
                    radii[i] = max(0, radii[i] - reduction)
                    radii[j] = max(0, radii[j] - reduction)

        if max_violation < 1e-14:
            break

    return centers, radii


def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Strategy 1: Hexagonal lattice with various patterns
    hex_patterns = [
        [5, 6, 5, 5, 5],  # 26
        [5, 5, 6, 5, 5],  # 26
        [6, 5, 5, 5, 5],  # 26
        [5, 5, 5, 6, 5],  # 26
        [4, 6, 5, 6, 5],  # 26
        [5, 6, 5, 6, 4],  # 26
        [6, 5, 6, 5, 4],  # 26
    ]

    for pattern_idx, pattern in enumerate(hex_patterns):
        if sum(pattern) != n:
            continue

        for seed in range(5):
            np.random.seed(seed)
            base_r = 0.09 + seed * 0.005

            centers = hexagonal_init(pattern, base_r)
            radii = np.ones(n) * base_r

            # Add small random perturbation
            centers += 0.003 * np.random.randn(n, 2)

            # Force optimization
            centers = force_optimize(centers, radii, n, steps=15000, base_lr=0.0015)

            # Compute optimal radii for these positions
            radii = compute_max_radii_from_positions(centers)

            # Gradient optimization
            centers, radii = gradient_optimize(centers, radii, n, max_iter=3000)

            # Recompute radii
            radii = compute_max_radii_from_positions(centers)

            # Final force optimization with fixed radii
            centers = force_optimize(centers, radii, n, steps=10000, base_lr=0.001)

            # Validate and fix
            centers, radii = validate_and_fix(centers, radii, n)

            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

    # Strategy 2: Start from square grid and perturb
    for grid_cols in [5, 6, 7]:
        if grid_cols > n:
            continue
        grid_rows = n // grid_cols + (1 if n % grid_cols > 0 else 0)
        # Only use first n circles from grid
        needed = n
        actual_cols = min(grid_cols, needed)
        actual_rows = (needed + actual_cols - 1) // actual_cols

        for seed in range(3):
            np.random.seed(seed + 100)
            base_r = 1.0 / (2 * max(actual_cols, actual_rows)) * 0.95

            centers = square_grid_init(actual_cols, actual_rows, base_r)
            if centers.shape[0] > n:
                centers = centers[:n]
            radii = np.ones(n) * base_r

            centers += 0.002 * np.random.randn(n, 2)

            centers = force_optimize(centers, radii, n, steps=15000, base_lr=0.0015)
            radii = compute_max_radii_from_positions(centers)
            centers, radii = gradient_optimize(centers, radii, n, max_iter=3000)
            radii = compute_max_radii_from_positions(centers)
            centers = force_optimize(centers, radii, n, steps=10000, base_lr=0.001)
            centers, radii = validate_and_fix(centers, radii, n)

            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

    # Strategy 3: Grow radii iteratively from a valid small packing
    for seed in range(3):
        np.random.seed(seed + 200)
        pattern = hex_patterns[0]
        base_r = 0.05

        centers = hexagonal_init(pattern, base_r)
        radii = np.ones(n) * base_r

        # Grow radii gradually
        for growth_step in range(100):
            # Recompute max radii
            radii = compute_max_radii_from_positions(centers)

            # Slightly increase target
            target_radii = radii * 1.01

            # Force optimize with target radii
            centers = force_optimize(centers, target_radii, n, steps=3000, base_lr=0.001)

        radii = compute_max_radii_from_positions(centers)
        centers, radii = gradient_optimize(centers, radii, n, max_iter=3000)
        radii = compute_max_radii_from_positions(centers)
        centers = force_optimize(centers, radii, n, steps=10000, base_lr=0.001)
        centers, radii = validate_and_fix(centers, radii, n)

        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Final validation
    best_centers, best_radii = validate_and_fix(best_centers, best_radii, n)

    # Ensure no negative radii
    best_radii = np.maximum(best_radii, 1e-10)

    return best_centers, best_radii, np.sum(best_radii)