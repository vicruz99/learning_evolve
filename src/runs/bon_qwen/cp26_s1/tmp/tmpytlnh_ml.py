import numpy as np
import cvxpy as cp


def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    r = cp.Variable(n)
    objective = cp.Maximize(cp.sum(r))
    constraints = []

    for i in range(n):
        constraints += [
            r[i] <= centers[i, 0],
            r[i] <= 1 - centers[i, 0],
            r[i] <= centers[i, 1],
            r[i] <= 1 - centers[i, 1],
            r[i] >= 0
        ]

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            constraints += [r[i] + r[j] <= dist]

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS, verbose=False)

    if problem.status in ["optimal", "optimal_inaccurate"]:
        return r.value
    return None


def run_packing():
    n = 26

    best_sum = 0
    best_centers = None
    best_radii = None

    # Try multiple initializations with different perturbations
    for seed in range(15):
        np.random.seed(seed * 37 + 11)

        # Initialize with hexagonal grid pattern
        centers = np.zeros((n, 2))

        # Hexagonal arrangement: 6 rows
        row_configs = [5, 5, 5, 5, 4, 2]
        idx = 0
        y_spacing = 1.0 / 7
        x_spacing = 1.0 / 6

        for row in range(6):
            y = (row + 0.5) * y_spacing
            offset = x_spacing / 2 if row % 2 == 1 else 0
            for col in range(row_configs[row]):
                x = (col + 0.5) * x_spacing + offset
                centers[idx] = [x, y] + np.random.randn(2) * 0.005
                idx += 1

        # Ensure centers are in valid range
        centers = np.clip(centers, 0.01, 0.99)

        # Local optimization using coordinate descent
        for iteration in range(150):
            radii = solve_radii_lp(centers)
            if radii is None:
                break

            current_sum = np.sum(radii)

            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()

            # Perturb each center with decreasing step size
            sigma = 0.015 / (1 + iteration * 0.015)

            for i in range(n):
                for _ in range(5):
                    delta = np.random.randn(2) * sigma
                    new_center = np.clip(centers[i] + delta, 0.005, 0.995)

                    old = centers[i].copy()
                    centers[i] = new_center

                    new_radii = solve_radii_lp(centers)
                    if new_radii is not None:
                        new_sum = np.sum(new_radii)
                        if new_sum > current_sum + 1e-10:
                            current_sum = new_sum
                            radii = new_radii
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_centers = centers.copy()
                                best_radii = radii.copy()
                        else:
                            centers[i] = old
                    else:
                        centers[i] = old

    # Final validation and return
    if best_centers is None:
        # Fallback: uniform grid
        best_centers = np.zeros((n, 2))
        best_radii = np.zeros(n)
        idx = 0
        for i in range(6):
            for j in range(5):
                if idx < n:
                    best_centers[idx] = [(j + 0.5) / 5, (i + 0.5) / 6]
                    best_radii[idx] = 0.05
                    idx += 1

    return best_centers, best_radii, best_sum