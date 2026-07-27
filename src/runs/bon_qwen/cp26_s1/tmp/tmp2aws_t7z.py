import numpy as np
from scipy.optimize import minimize


def compute_constraints(params, n):
    """Compute all constraint values for the optimization."""
    centers = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    constraints_list = []

    # Boundary constraints: r_i <= x_i <= 1 - r_i and r_i <= y_i <= 1 - r_i
    for i in range(n):
        r = radii[i]
        x = centers[i, 0]
        y = centers[i, 1]
        constraints_list.append(x - r)
        constraints_list.append(1 - x - r)
        constraints_list.append(y - r)
        constraints_list.append(1 - y - r)

    # Overlap constraints: ||c_i - c_j|| >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            constraints_list.append(dist - radii[i] - radii[j])

    return np.array(constraints_list)


def make_constraint_function(n):
    """Create the constraint function for scipy."""
    def constraint_fun(params):
        return compute_constraints(params, n)
    return constraint_fun


def generate_initial_config(n, r_init, row_counts):
    """Generate an initial hexagonal configuration."""
    centers = []
    for row_idx, row_size in enumerate(row_counts):
        y = r_init + row_idx * r_init * np.sqrt(3)

        if row_idx % 2 == 0:
            start_x = r_init
        else:
            start_x = 2 * r_init

        for col in range(row_size):
            x = start_x + col * 2 * r_init
            if x <= 1 - r_init:
                centers.append([x, y])

    return np.array(centers)


def run_packing():
    n = 26

    # Define objective: maximize sum of radii (minimize negative sum)
    def objective(params):
        radii = params[2 * n:]
        return -np.sum(radii)

    # Define constraint function
    constraint_fun = make_constraint_function(n)
    constraints = {'type': 'ineq', 'fun': constraint_fun}

    # Bounds for all variables
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0)])  # x, y in [0, 1]
        bounds.append((0.0, 0.5))  # radius in [0, 0.5]

    # Try multiple initial configurations
    best_params = None
    best_obj = np.inf
    best_sum_radii = 0.0

    # Different row distributions that sum to 26
    row_distributions = [
        [6, 5, 5, 5, 5],
        [5, 6, 5, 5, 5],
        [5, 5, 6, 5, 5],
        [5, 5, 5, 6, 5],
        [5, 5, 5, 5, 6],
        [7, 5, 5, 5, 4],
        [4, 5, 5, 5, 7],
        [6, 6, 5, 5, 4],
        [5, 5, 7, 5, 4],
    ]

    initial_radii_list = [0.075, 0.08, 0.085, 0.09]

    for r_init in initial_radii_list:
        for row_dist in row_distributions:
            if sum(row_dist) != n:
                continue

            centers_init = generate_initial_config(n, r_init, row_dist)
            if len(centers_init) != n:
                continue

            radii_init = np.ones(n) * r_init
            params_init = np.concatenate([centers_init.flatten(), radii_init])

            # Check if initial config is valid
            init_constraints = compute_constraints(params_init, n)
            if np.min(init_constraints) < -0.01:
                # Some initial circles might overlap or be out of bounds,
                # but optimizer should fix this
                pass

            try:
                result = minimize(
                    objective,
                    params_init,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={
                        'maxiter': 2000,
                        'ftol': 1e-12,
                        'disp': False
                    }
                )

                if result.fun < best_obj:
                    # Verify the result
                    params = result.x
                    centers = params[:2 * n].reshape(n, 2)
                    radii = params[2 * n:]

                    # Check constraints
                    final_constraints = compute_constraints(params, n)
                    if np.min(final_constraints) >= -1e-6:
                        best_obj = result.fun
                        best_params = params.copy()
                        best_sum_radii = np.sum(radii)
            except Exception:
                continue

    if best_params is not None:
        centers = best_params[:2 * n].reshape(n, 2)
        radii = best_params[2 * n:]
        radii = np.maximum(radii, 0)  # Ensure non-negative
        return centers, radii, np.sum(radii)

    # Fallback: generate a valid but simple configuration
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.07
    idx = 0
    for row in range(6):
        for col in range(5):
            if idx < n:
                centers[idx, 0] = 0.1 + col * 0.2
                centers[idx, 1] = 0.1 + row * 0.17
                idx += 1

    return centers, radii, np.sum(radii)