# sol_000281 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9d8cea89) state=20178fd4 sum of radii=2.625734 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(params):
    """Maximize sum of radii by minimizing negative sum."""
    return -np.sum(params[2::3])

def boundary_constraints(params):
    """Each circle must be within the unit square with non-negative radius."""
    n = N_CIRCLES
    result = np.zeros(5 * n)
    for i in range(n):
        x = params[3 * i]
        y = params[3 * i + 1]
        r = params[3 * i + 2]
        result[5 * i] = x - r
        result[5 * i + 1] = 1.0 - x - r
        result[5 * i + 2] = y - r
        result[5 * i + 3] = 1.0 - y - r
        result[5 * i + 4] = r
    return result

def non_overlap_constraints(params):
    """Distance between any two circle centers must be >= sum of their radii."""
    n = N_CIRCLES
    n_pairs = n * (n - 1) // 2
    result = np.zeros(n_pairs)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi = params[3 * i]
            yi = params[3 * i + 1]
            ri = params[3 * i + 2]
            xj = params[3 * j]
            yj = params[3 * j + 1]
            rj = params[3 * j + 2]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx * dx + dy * dy
            result[idx] = dist_sq - (ri + rj) ** 2
            idx += 1
    return result

def make_hexagonal_config(spacing, radius, row_offset=0.0):
    """Create a hexagonal grid configuration."""
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    idx = 0
    sqrt3 = np.sqrt(3)
    for row in range(6):
        n_cols = 5 if row % 2 == 0 else 4
        for col in range(n_cols):
            if idx >= n:
                break
            x = radius + col * spacing
            y = radius + row * spacing * sqrt3 / 2.0 + row_offset
            if row % 2 == 1:
                x += spacing / 2.0
            centers[idx] = [x, y]
            idx += 1
    radii = np.ones(n) * radius
    return centers, radii

def make_square_config(spacing, radius):
    """Create a square grid configuration."""
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(6):
        n_cols = 5 if row < 5 else 1
        for col in range(n_cols):
            if idx >= n:
                break
            centers[idx] = [radius + col * spacing, radius + row * spacing]
            idx += 1
    radii = np.ones(n) * radius
    return centers, radii

def make_random_config(seed):
    """Create a random configuration."""
    n = N_CIRCLES
    np.random.seed(seed)
    centers = np.random.rand(n, 2) * 0.7 + 0.15
    radii = np.ones(n) * 0.05
    return centers, radii

def params_to_arrays(params):
    """Convert flattened params to centers and radii arrays."""
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = params[3 * i]
        centers[i, 1] = params[3 * i + 1]
        radii[i] = params[3 * i + 2]
    return centers, radii

def arrays_to_params(centers, radii):
    """Convert centers and radii arrays to flattened params."""
    n = N_CIRCLES
    params = np.zeros(3 * n)
    for i in range(n):
        params[3 * i] = centers[i, 0]
        params[3 * i + 1] = centers[i, 1]
        params[3 * i + 2] = radii[i]
    return params

def optimize_from(params0):
    """Run SLSQP optimization from given initial parameters."""
    n = N_CIRCLES
    cons = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': non_overlap_constraints}
    ]
    bounds = [(0.0, 1.0)] * (3 * n)
    for i in range(n):
        bounds[3 * i + 2] = (0.0, 0.5)
    result = minimize(
        objective, params0, method='SLSQP',
        bounds=bounds, constraints=cons,
        options={'maxiter': 2000, 'ftol': 1e-14, 'disp': False}
    )
    return result

def run_packing():
    global N_CIRCLES
    N_CIRCLES = 26
    n = N_CIRCLES

    best_centers = None
    best_radii = None
    best_sum = 0.0

    # List of initial configurations to try
    initial_configs = []

    # Hexagonal configurations with different parameters
    for spacing in [0.18, 0.19, 0.20]:
        for radius in [0.09, 0.095, 0.10]:
            centers, radii = make_hexagonal_config(spacing, radius)
            params = arrays_to_params(centers, radii)
            initial_configs.append(params)

    # Hexagonal with offset
    for spacing in [0.185]:
        for radius in [0.095]:
            for offset in [0.02, -0.02]:
                centers, radii = make_hexagonal_config(spacing, radius, row_offset=offset)
                params = arrays_to_params(centers, radii)
                initial_configs.append(params)

    # Square configurations
    for spacing in [0.18, 0.19]:
        for radius in [0.09, 0.095]:
            centers, radii = make_square_config(spacing, radius)
            params = arrays_to_params(centers, radii)
            initial_configs.append(params)

    # Random configurations
    for seed in range(10):
        centers, radii = make_random_config(seed)
        params = arrays_to_params(centers, radii)
        initial_configs.append(params)

    # Optimize from each initial configuration
    for params0 in initial_configs:
        result = optimize_from(params0)

        if not result.success and result.fun < -best_sum - 0.0001:
            pass  # Still check even if not successful

        radii_arr = result.x[2::3]
        current_sum = np.sum(radii_arr)

        # Check feasibility with tolerance
        feasible = True
        centers_arr = np.array([[result.x[3 * i], result.x[3 * i + 1]] for i in range(n)])

        for i in range(n):
            x, y = centers_arr[i]
            r = radii_arr[i]
            if x - r < -1e-8 or x + r > 1 + 1e-8 or y - r < -1e-8 or y + r > 1 + 1e-8 or r < -1e-8:
                feasible = False
                break

        if feasible:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt((centers_arr[i, 0] - centers_arr[j, 0]) ** 2 +
                                   (centers_arr[i, 1] - centers_arr[j, 1]) ** 2)
                    if dist < radii_arr[i] + radii_arr[j] - 1e-8:
                        feasible = False
                        break
                if not feasible:
                    break

        if feasible and current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers_arr.copy()
            best_radii = radii_arr.copy()

    # If no feasible solution found, use the best objective regardless
    if best_centers is None:
        # Fallback: run one more optimization with a good hexagonal start
        centers, radii = make_hexagonal_config(0.19, 0.095)
        params0 = arrays_to_params(centers, radii)
        result = optimize_from(params0)
        best_centers = np.array([[result.x[3 * i], result.x[3 * i + 1]] for i in range(n)])
        best_radii = result.x[2::3].copy()
        best_sum = np.sum(best_radii)

    # Final cleanup: ensure radii are non-negative
    best_radii = np.maximum(best_radii, 0.0)

    # Ensure centers are within bounds
    for i in range(n):
        best_centers[i, 0] = np.clip(best_centers[i, 0], best_radii[i], 1.0 - best_radii[i])
        best_centers[i, 1] = np.clip(best_centers[i, 1], best_radii[i], 1.0 - best_radii[i])

    return best_centers, best_radii, np.sum(best_radii)
