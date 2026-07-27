import numpy as np
from scipy.optimize import minimize


def compute_penalty(centers, radii, n):
    """Compute penalty for constraint violations."""
    penalty = 0.0

    # Overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += overlap * overlap

    # Boundary penalty
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0:
            penalty += (x - r) * (x - r)
        if x + r > 1:
            penalty += (x + r - 1) * (x + r - 1)
        if y - r < 0:
            penalty += (y - r) * (y - r)
        if y + r > 1:
            penalty += (y + r - 1) * (y + r - 1)

    return penalty


def objective(x, n):
    """Objective: maximize sum of radii with penalty for violations."""
    centers = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    obj = -np.sum(radii)
    penalty = compute_penalty(centers, radii, n)

    return obj + 1000 * penalty


def objective_smooth(x, n):
    """Smooth objective with differentiable penalty."""
    centers = x[:2 * n].reshape(n, 2)
    radii = x[2 * n:]

    obj = -np.sum(radii)
    penalty = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy + 1e-16)
            overlap = radii[i] + radii[j] - dist
            if overlap > 0:
                penalty += overlap * overlap

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        penalty += max(0, r - x) * max(0, r - x)
        penalty += max(0, x + r - 1) * max(0, x + r - 1)
        penalty += max(0, r - y) * max(0, r - y)
        penalty += max(0, y + r - 1) * max(0, y + r - 1)

    return obj + 2000 * penalty


def init_hexagonal_grid(n):
    """Initialize circles in a hexagonal pattern."""
    centers = np.zeros((n, 2))

    # For n=26: use rows of 6,5,6,5,4
    row_counts = [6, 5, 6, 5, 4]
    n_rows = len(row_counts)
    max_cols = max(row_counts)

    spacing_x = 0.92 / max_cols
    spacing_y = 0.92 / (n_rows - 0.4)

    idx = 0
    for row in range(n_rows):
        count = row_counts[row]

        if row % 2 == 0:
            x_start = 0.04
        else:
            x_start = 0.04 + spacing_x / 2

        y = 0.04 + row * spacing_y

        for col in range(count):
            if idx >= n:
                break
            x = x_start + col * spacing_x
            centers[idx] = [x, y]
            idx += 1

    radii = np.ones(n) * 0.04
    return centers, radii


def init_random_perturbed(n, seed):
    """Initialize with random perturbation of hexagonal grid."""
    np.random.seed(seed)
    centers, radii = init_hexagonal_grid(n)
    centers += np.random.randn(n, 2) * 0.02
    centers = np.clip(centers, 0.05, 0.95)
    radii = np.ones(n) * 0.035 + np.random.rand(n) * 0.01
    return centers, radii


def force_expand(centers, radii, n, iterations=5000):
    """Expand circles using repulsive forces."""
    lr = 0.005
    expansion = 0.00005

    for it in range(iterations):
        forces = np.zeros((n, 2))

        # Repulsion between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)

                if dist < radii[i] + radii[j] + 0.02 and dist > 1e-10:
                    overlap = max(0, radii[i] + radii[j] - dist)
                    force_mag = overlap * 50
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]

        # Boundary repulsion
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0:
                forces[i, 0] += (r - x) * 50
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 50
            if y - r < 0:
                forces[i, 1] += (r - y) * 50
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 50

        centers += lr * forces

        # Project to feasible
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            centers[i, 0] = np.clip(x, r, 1 - r)
            centers[i, 1] = np.clip(y, r, 1 - r)

        # Expand
        radii += expansion

    return centers, radii


def ensure_feasibility(centers, radii, n):
    """Ensure circles don't overlap and stay inside square."""
    for iteration in range(300):
        changed = False

        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)

                if dist < radii[i] + radii[j]:
                    avg = max(1e-10, dist / 2)
                    if radii[i] > avg:
                        radii[i] = avg
                        changed = True
                    if radii[j] > avg:
                        radii[j] = avg
                        changed = True

        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            max_r = min(x, 1 - x, y, 1 - y)
            if r > max_r:
                radii[i] = max(0, max_r)
                changed = True

        if not changed:
            break

    return centers, radii


def optimize_packing(centers_init, radii_init, n):
    """Optimize packing using gradient-based method."""
    x0 = np.concatenate([centers_init.flatten(), radii_init])

    bounds = []
    for i in range(2 * n):
        bounds.append((0.005, 0.995))
    for i in range(n):
        bounds.append((0.001, 0.5))

    result = minimize(objective_smooth, x0, method='L-BFGS-B',
                      bounds=bounds, args=(n,),
                      options={'maxiter': 50000, 'ftol': 1e-16, 'gtol': 1e-12})

    centers_opt = result.x[:2 * n].reshape(n, 2)
    radii_opt = result.x[2 * n:]

    return centers_opt, radii_opt


def run_packing():
    n = 26

    best_sum = 0
    best_centers = None
    best_radii = None

    # Strategy 1: Hexagonal grid + force expansion + optimization
    centers, radii = init_hexagonal_grid(n)
    centers, radii = force_expand(centers, radii, n, iterations=8000)
    centers, radii = optimize_packing(centers, radii, n)
    centers, radii = ensure_feasibility(centers, radii, n)
    current_sum = np.sum(radii)

    if current_sum > best_sum:
        best_sum = current_sum
        best_centers = centers.copy()
        best_radii = radii.copy()

    # Strategy 2: Multiple random restarts
    for seed in range(30):
        centers, radii = init_random_perturbed(n, seed)
        centers, radii = force_expand(centers, radii, n, iterations=6000)
        centers, radii = optimize_packing(centers, radii, n)
        centers, radii = ensure_feasibility(centers, radii, n)

        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Strategy 3: Start with tighter initial radii and expand
    for seed in range(20):
        np.random.seed(seed + 1000)
        centers, radii = init_hexagonal_grid(n)
        radii = np.ones(n) * 0.02
        centers += np.random.randn(n, 2) * 0.01
        centers = np.clip(centers, 0.1, 0.9)

        centers, radii = force_expand(centers, radii, n, iterations=10000)
        centers, radii = optimize_packing(centers, radii, n)
        centers, radii = ensure_feasibility(centers, radii, n)

        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Final refinement with very small steps
    best_centers, best_radii = optimize_packing(best_centers, best_radii, n)
    best_centers, best_radii = ensure_feasibility(best_centers, best_radii, n)

    return best_centers, best_radii, np.sum(best_radii)