import numpy as np
from scipy.optimize import minimize
import math

def run_packing():
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0

    # Strategy 1: Hexagonal initializations with multiple offsets
    for seed in range(8):
        centers, radii = init_hexagonal(n, seed)
        result = optimize_slsqp(n, centers, radii)
        if result is not None:
            c, r, s = result
            if s > best_sum:
                best_sum = s
                best_centers = c.copy()
                best_radii = r.copy()

    # Strategy 2: Random initializations
    for seed in range(15):
        centers, radii = init_random(n, seed)
        result = optimize_slsqp(n, centers, radii)
        if result is not None:
            c, r, s = result
            if s > best_sum:
                best_sum = s
                best_centers = c.copy()
                best_radii = r.copy()

    # Strategy 3: Corner-focused initialization
    for seed in range(5):
        centers, radii = init_corner_focused(n, seed)
        result = optimize_slsqp(n, centers, radii)
        if result is not None:
            c, r, s = result
            if s > best_sum:
                best_sum = s
                best_centers = c.copy()
                best_radii = r.copy()

    # Strategy 4: Force-based growing circles from best so far
    if best_centers is not None:
        centers, radii = force_based_optimize(n, best_centers, best_radii)
        s = np.sum(radii)
        if s > best_sum:
            best_sum = s
            best_centers = centers.copy()
            best_radii = radii.copy()
        # Also try optimizing the force-based result with SLSQP
        result = optimize_slsqp(n, centers, radii)
        if result is not None:
            c, r, s = result
            if s > best_sum:
                best_sum = s
                best_centers = c.copy()
                best_radii = r.copy()

    # Final perturbation search
    if best_centers is not None:
        for seed in range(10):
            np.random.seed(seed + 100)
            c = best_centers.copy()
            r = best_radii.copy()
            # Add small perturbation
            c += np.random.normal(0, 0.005, c.shape)
            c[:, 0] = np.clip(c[:, 0], 0.01, 0.99)
            c[:, 1] = np.clip(c[:, 1], 0.01, 0.99)
            r += np.random.normal(0, 0.002, r.shape)
            r = np.maximum(r, 0.01)
            result = optimize_slsqp(n, c, r)
            if result is not None:
                cc, rr, s = result
                if s > best_sum:
                    best_sum = s
                    best_centers = cc.copy()
                    best_radii = rr.copy()

    if best_centers is None:
        best_centers, best_radii = init_hexagonal(n, 0)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum


def init_hexagonal(n, seed=0):
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09)

    np.random.seed(seed)
    r = 0.09 + np.random.uniform(0, 0.01)
    h_spacing = 2 * r * (0.95 + np.random.uniform(0, 0.1))
    v_spacing = math.sqrt(3) * r * (0.95 + np.random.uniform(0, 0.1))

    rows = [5, 4, 5, 4, 5, 3]
    idx = 0
    base_x = 0.05 + np.random.uniform(0, 0.05)
    base_y = 0.05 + np.random.uniform(0, 0.05)

    for row_i, n_cols in enumerate(rows):
        y = base_y + row_i * v_spacing
        x_offset = base_x if row_i % 2 == 0 else base_x + h_spacing / 2

        for col in range(n_cols):
            if idx >= n:
                break
            x = x_offset + col * h_spacing
            centers[idx] = [np.clip(x, r + 0.01, 1 - r - 0.01),
                           np.clip(y, r + 0.01, 1 - r - 0.01)]
            idx += 1

    return centers, radii


def init_grid(n, seed=0):
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09)

    np.random.seed(seed)
    idx = 0
    spacing_x = 0.17 + np.random.uniform(0, 0.03)
    spacing_y = 0.14 + np.random.uniform(0, 0.03)
    base_x = 0.08 + np.random.uniform(0, 0.04)
    base_y = 0.08 + np.random.uniform(0, 0.04)

    for row in range(6):
        for col in range(5):
            if idx >= n:
                break
            x = base_x + col * spacing_x
            y = base_y + row * spacing_y
            centers[idx] = [np.clip(x, 0.02, 0.98), np.clip(y, 0.02, 0.98)]
            idx += 1
        if idx >= n:
            break

    return centers, radii


def init_corner_focused(n, seed=0):
    centers = np.zeros((n, 2))
    radii = np.zeros(n)

    np.random.seed(seed)
    idx = 0

    # Place some circles near corners (can be larger)
    corner_positions = [
        [0.13, 0.13], [0.87, 0.13],
        [0.13, 0.87], [0.87, 0.87]
    ]
    for pos in corner_positions:
        centers[idx] = pos
        radii[idx] = 0.13
        idx += 1

    # Fill remaining with hexagonal-ish pattern
    r = 0.085
    h_spacing = 2 * r
    v_spacing = math.sqrt(3) * r

    row = 0
    while idx < n:
        n_cols = 5 if row % 2 == 0 else 4
        for col in range(n_cols):
            if idx >= n:
                break
            x = 0.08 + col * h_spacing + (h_spacing / 2 if row % 2 == 1 else 0)
            y = 0.08 + row * v_spacing
            centers[idx] = [np.clip(x, r + 0.01, 1 - r - 0.01),
                           np.clip(y, r + 0.01, 1 - r - 0.01)]
            radii[idx] = r
            idx += 1
        row += 1

    return centers, radii


def init_random(n, seed=0):
    np.random.seed(seed)
    centers = np.random.uniform(0.15, 0.85, (n, 2))
    radii = np.full(n, 0.08)
    return centers, radii


def optimize_slsqp(n, centers, radii):
    x0 = np.concatenate([centers.flatten(), radii])

    def objective(x):
        return -np.sum(x[2 * n:])

    bounds = [(0.0001, 0.9999)] * (2 * n) + [(0.0001, 0.5)] * n

    def constraint_func(x):
        c = x[:2 * n].reshape((n, 2))
        r = x[2 * n:]

        cons = []

        # Boundary constraints
        for i in range(n):
            cons.append(c[i, 0] - r[i])
            cons.append(1.0 - c[i, 0] - r[i])
            cons.append(c[i, 1] - r[i])
            cons.append(1.0 - c[i, 1] - r[i])

        # Non-overlap (squared distance to avoid sqrt)
        for i in range(n):
            for j in range(i + 1, n):
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                dist_sq = dx * dx + dy * dy
                r_sum = r[i] + r[j]
                cons.append(dist_sq - r_sum * r_sum)

        return np.array(cons)

    cons = {'type': 'ineq', 'fun': constraint_func}

    try:
        result = minimize(objective, x0, bounds=bounds, constraints=cons,
                         method='SLSQP',
                         options={'maxiter': 5000, 'ftol': 1e-15, 'disp': False})

        if result.fun < -2.0:
            centers_opt = result.x[:2 * n].reshape((n, 2))
            radii_opt = result.x[2 * n:]
            s = np.sum(radii_opt)

            if check_feasibility(centers_opt, radii_opt):
                return centers_opt, radii_opt, s
    except Exception:
        pass

    return None


def force_based_optimize(n, centers, radii):
    centers = centers.copy()
    radii = radii.copy()

    # Phase 1: Expand radii uniformly while adjusting positions
    for step in range(5000):
        # Slightly increase all radii
        expansion_rate = 1.0 + 0.0003 * (1.0 - step / 10000.0)
        radii *= expansion_rate

        # Resolve overlaps by pushing circles apart
        for _ in range(5):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[j, 0] - centers[i, 0]
                    dy = centers[j, 1] - centers[i, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]

                    if dist < min_dist and dist > 1e-12:
                        overlap = (min_dist - dist) / 2.0
                        nx = dx / dist
                        ny = dy / dist
                        centers[i, 0] -= overlap * nx
                        centers[i, 1] -= overlap * ny
                        centers[j, 0] += overlap * nx
                        centers[j, 1] += overlap * ny

        # Clamp to boundaries
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i] + 1e-8, 1.0 - radii[i] - 1e-8)
            centers[i, 1] = np.clip(centers[i, 1], radii[i] + 1e-8, 1.0 - radii[i] - 1e-8)

    # Phase 2: Individual radius optimization
    for _ in range(20):
        for i in range(n):
            # Find maximum possible radius for circle i
            max_r = min(centers[i, 0], 1 - centers[i, 0],
                       centers[i, 1], 1 - centers[i, 1])
            for j in range(n):
                if i != j:
                    dist = math.sqrt((centers[i, 0] - centers[j, 0]) ** 2 +
                                    (centers[i, 1] - centers[j, 1]) ** 2)
                    max_r = min(max_r, dist - radii[j])
            radii[i] = max(radii[i], max_r - 1e-8)

        # Small position adjustments
        for _ in range(10):
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers[j, 0] - centers[i, 0]
                    dy = centers[j, 1] - centers[i, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]

                    if dist < min_dist and dist > 1e-12:
                        overlap = (min_dist - dist) / 2.0
                        nx = dx / dist
                        ny = dy / dist
                        centers[i, 0] -= overlap * nx * 0.5
                        centers[i, 1] -= overlap * ny * 0.5
                        centers[j, 0] += overlap * nx * 0.5
                        centers[j, 1] += overlap * ny * 0.5

            for i in range(n):
                centers[i, 0] = np.clip(centers[i, 0], radii[i] + 1e-8, 1.0 - radii[i] - 1e-8)
                centers[i, 1] = np.clip(centers[i, 1], radii[i] + 1e-8, 1.0 - radii[i] - 1e-8)

    return centers, radii


def check_feasibility(centers, radii):
    n = len(radii)
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-10 or x + r > 1.0 + 1e-10 or y - r < -1e-10 or y + r > 1.0 + 1e-10:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((centers[i, 0] - centers[j, 0]) ** 2 +
                           (centers[i, 1] - centers[j, 1]) ** 2)
            if dist < radii[i] + radii[j] - 1e-10:
                return False

    return True