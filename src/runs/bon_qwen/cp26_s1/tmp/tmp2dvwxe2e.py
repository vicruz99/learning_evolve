import numpy as np
from scipy.optimize import minimize


def compute_penalty(centers, radii, weight):
    """Compute constraint violation penalty."""
    n = len(radii)
    penalty = 0.0

    # Boundary constraints: circle must be inside [0,1]x[0,1]
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r > x:
            penalty += weight * (r - x) ** 2
        if r > 1 - x:
            penalty += weight * (r - (1 - x)) ** 2
        if r > y:
            penalty += weight * (r - y) ** 2
        if r > 1 - y:
            penalty += weight * (r - (1 - y)) ** 2

    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if min_dist > dist:
                penalty += weight * (min_dist - dist) ** 2

    return penalty


def objective(params, n, weight):
    """Objective function: maximize sum of radii."""
    centers = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]

    obj = -np.sum(radii)
    obj += compute_penalty(centers, radii, weight)

    return obj


def get_bounds(n):
    """Get variable bounds."""
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.999))
        bounds.append((0.001, 0.5))
    return bounds


def generate_hexagonal_grid(n):
    """Generate initial positions on a hexagonal grid."""
    centers = np.zeros((n, 2))
    idx = 0
    row = 0
    spacing_x = 0.16
    spacing_y = 0.14
    offset_x = 0.08
    offset_y = 0.1

    while idx < n:
        cols = 6 if row % 2 == 0 else 5
        for col in range(cols):
            if idx >= n:
                break
            x = offset_x + col * spacing_x
            y = offset_y + row * spacing_y
            if row % 2 == 1:
                x += offset_x
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            centers[idx] = [x, y]
            idx += 1
        row += 1

    return centers


def generate_corner_heavy(n):
    """Generate initialization favoring larger circles in corners."""
    centers = np.array([
        [0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88],
        [0.5, 0.12], [0.5, 0.88], [0.12, 0.5], [0.88, 0.5],
        [0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75],
        [0.33, 0.33], [0.67, 0.33], [0.33, 0.67], [0.67, 0.67],
        [0.5, 0.5],
        [0.2, 0.35], [0.8, 0.35], [0.2, 0.65], [0.8, 0.65],
        [0.4, 0.2], [0.6, 0.2], [0.4, 0.8], [0.6, 0.8],
        [0.3, 0.5], [0.7, 0.5],
    ])
    return centers[:n]


def ensure_feasibility(centers, radii, n):
    """Post-process to ensure strict feasibility."""
    radii_new = radii.copy()
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y) - 1e-10
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                max_r = min(max_r, dist - radii_new[j] - 1e-10)
        radii_new[i] = min(radii_new[i], max(0, max_r))
    return radii_new


def run_packing():
    n = 26

    best_sum = 0
    best_centers = None
    best_radii = None

    num_restarts = 20
    for restart in range(num_restarts):
        # Choose initialization strategy
        if restart == 0:
            centers = generate_hexagonal_grid(n)
            radii = np.full(n, 0.05)
        elif restart == 1:
            centers = generate_corner_heavy(n)
            radii = np.array([0.08] * 4 + [0.065] * 4 + [0.055] * 4 + [0.05] * 8 + [0.045] * 6)
        elif restart == 2:
            # Uniform grid-like
            centers = np.zeros((n, 2))
            for i in range(n):
                row = i // 6
                col = i % 6
                x = 0.1 + col * 0.15
                y = 0.1 + row * 0.15
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                centers[i] = [x, y]
            radii = np.full(n, 0.06)
        else:
            # Random with some structure
            np.random.seed(restart * 42)
            centers = np.random.rand(n, 2) * 0.7 + 0.15
            # Sort by x-coordinate to avoid clustering
            centers = centers[centers[:, 0].argsort()]
            radii = np.full(n, 0.04)

        params = np.concatenate([centers.flatten(), radii])
        bounds = get_bounds(n)

        # Annealing: gradually increase penalty weight
        for weight in [50, 200, 1000, 5000, 20000]:
            result = minimize(
                lambda p, w=weight: objective(p, n, w),
                params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            params = result.x

        # Extract results
        centers_opt = params[:2 * n].reshape(n, 2)
        radii_opt = params[2 * n:].copy()

        # Ensure strict feasibility
        radii_opt = ensure_feasibility(centers_opt, radii_opt, n)

        # Calculate sum
        s = np.sum(radii_opt)
        if s > best_sum:
            best_sum = s
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()

    return best_centers, best_radii, best_sum