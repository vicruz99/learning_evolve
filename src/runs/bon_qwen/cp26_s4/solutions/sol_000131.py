# sol_000131 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state da2150ba) state=532e10d8 sum of radii=2.617858 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math


def softplus(x):
    """Smooth approximation of max(0, x), numerically stable."""
    x = float(x)
    if x > 20:
        return x
    if x < -20:
        return 0.0
    return math.log1p(math.exp(-abs(x))) + max(x, 0.0)


def compute_objective_and_grad(x, n, penalty_weight):
    """Compute objective value and gradient for circle packing optimization."""
    c = x[:2 * n].reshape(n, 2)
    r = x[2 * n:]

    # Objective: maximize sum of radii (negate for minimization)
    obj = -np.sum(r)

    # Gradient of objective
    grad = np.zeros_like(x)
    grad[2 * n:] = -1.0

    # Smooth penalty for boundary violations
    pw = float(penalty_weight)
    for i in range(n):
        # x >= r constraint: violation when r - x > 0
        val = r[i] - c[i, 0]
        if val > 0:
            obj += pw * val * val
            grad[2 * n + i] += 2 * pw * val
            grad[2 * i] -= 2 * pw * val

        # x + r <= 1 constraint: violation when x + r - 1 > 0
        val = r[i] + c[i, 0] - 1.0
        if val > 0:
            obj += pw * val * val
            grad[2 * n + i] += 2 * pw * val
            grad[2 * i] += 2 * pw * val

        # y >= r constraint
        val = r[i] - c[i, 1]
        if val > 0:
            obj += pw * val * val
            grad[2 * n + i] += 2 * pw * val
            grad[2 * i + 1] -= 2 * pw * val

        # y + r <= 1 constraint
        val = r[i] + c[i, 1] - 1.0
        if val > 0:
            obj += pw * val * val
            grad[2 * n + i] += 2 * pw * val
            grad[2 * i + 1] += 2 * pw * val

    # Smooth penalty for pairwise overlaps
    for i in range(n):
        for j in range(i + 1, n):
            diff = c[i] - c[j]
            dist_sq = np.sum(diff * diff)
            dist = math.sqrt(dist_sq + 1e-12)

            val = r[i] + r[j] - dist
            if val > 0:
                obj += pw * val * val

                factor = 2 * pw * val
                grad[2 * n + i] += factor
                grad[2 * n + j] += factor

                # Gradient w.r.t. centers
                if dist > 1e-12:
                    dgrad = -factor / dist
                    grad[2 * i] += dgrad * diff[0]
                    grad[2 * i + 1] += dgrad * diff[1]
                    grad[2 * j] -= dgrad * diff[0]
                    grad[2 * j + 1] -= dgrad * diff[1]

    return obj, grad


def make_obj_func(n, pw):
    """Create objective function for given penalty weight."""
    def f(x):
        return compute_objective_and_grad(x, n, pw)[0]
    return f


def make_grad_func(n, pw):
    """Create gradient function for given penalty weight."""
    def g(x):
        return compute_objective_and_grad(x, n, pw)[1]
    return g


def make_obj_and_grad_func(n, pw):
    """Create objective and gradient function."""
    def fg(x):
        return compute_objective_and_grad(x, n, pw)
    return fg


def initialize_hexagonal(n, r_init):
    """Initialize circles in a hexagonal grid pattern."""
    centers_list = []
    y = float(r_init)
    row = 0

    while len(centers_list) < n:
        if row % 2 == 0:
            x_start = float(r_init)
        else:
            x_start = float(r_init) * 2.0

        col = 0
        while len(centers_list) < n:
            x = x_start + col * 2.0 * float(r_init)
            if x + float(r_init) > 1.0:
                break
            centers_list.append([x, y])
            col += 1

        y += float(r_init) * 1.732050808  # sqrt(3)
        row += 1

    centers_arr = np.array(centers_list[:n], dtype=np.float64)
    radii_arr = np.full(n, float(r_init), dtype=np.float64)

    if len(centers_list) < n:
        # Fill remaining with scattered positions
        rng = np.random.RandomState(123)
        for idx in range(n - len(centers_list)):
            x = rng.uniform(0.1, 0.9)
            y = rng.uniform(0.1, 0.9)
            centers_arr = np.vstack([centers_arr, [x, y]])
            radii_arr = np.append(radii_arr, 0.02)

    return centers_arr[:n], radii_arr[:n]


def enforce_feasibility(centers, radii, n):
    """Post-process to ensure all constraints are satisfied."""
    centers = centers.copy()
    radii = radii.copy()

    # Ensure boundary constraints
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1])
        radii[i] = min(radii[i], max(max_r, 0.0))

    # Fix overlaps by iterative shrinking
    for iteration in range(500):
        max_overlap = -1e-15
        max_i = -1
        max_j = -1

        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx * dx + dy * dy)
                overlap = radii[i] + radii[j] - dist
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_i = i
                    max_j = j

        if max_overlap < 1e-12:
            break

        # Shrink proportionally
        total_r = radii[max_i] + radii[max_j]
        if total_r > 0:
            shrink_i = max_overlap * radii[max_i] / total_r + 1e-10
            shrink_j = max_overlap * radii[max_j] / total_r + 1e-10
            radii[max_i] = max(0.0, radii[max_i] - shrink_i)
            radii[max_j] = max(0.0, radii[max_j] - shrink_j)
        else:
            radii[max_i] = 0.0
            radii[max_j] = 0.0

    return centers, radii


def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Try multiple initializations with different starting radii
    r_init_values = [0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09]

    for trial_idx, r_init in enumerate(r_init_values):
        np.random.seed(42 + trial_idx)

        centers_init, radii_init = initialize_hexagonal(n, r_init)
        x0 = np.concatenate([centers_init.flatten(), radii_init])

        # Bounds for optimization
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

        # Multi-stage optimization with increasing penalty weights
        x_current = x0.copy()
        penalty_stages = [50, 200, 800, 2000, 5000]

        for pw in penalty_stages:
            obj_fn = make_obj_func(n, pw)
            grad_fn = make_grad_func(n, pw)

            try:
                result = minimize(
                    obj_fn,
                    x_current,
                    method='L-BFGS-B',
                    jac=grad_fn,
                    bounds=bounds,
                    options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-14}
                )
                x_current = result.x
            except Exception:
                pass

        # Evaluate the result
        c_opt = x_current[:2 * n].reshape(n, 2)
        r_opt = x_current[2 * n:]

        # Enforce feasibility
        c_feas, r_feas = enforce_feasibility(c_opt, r_opt, n)

        sum_r = np.sum(r_feas)

        # Verify feasibility
        feasible = True
        for i in range(n):
            if r_feas[i] < 0:
                feasible = False
                break
            if c_feas[i, 0] - r_feas[i] < -1e-10 or c_feas[i, 0] + r_feas[i] > 1.0 + 1e-10:
                feasible = False
                break
            if c_feas[i, 1] - r_feas[i] < -1e-10 or c_feas[i, 1] + r_feas[i] > 1.0 + 1e-10:
                feasible = False
                break
        if feasible:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = c_feas[i, 0] - c_feas[j, 0]
                    dy = c_feas[i, 1] - c_feas[j, 1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < r_feas[i] + r_feas[j] - 1e-10:
                        feasible = False
                        break
                if not feasible:
                    break

        if feasible and sum_r > best_sum:
            best_sum = sum_r
            best_centers = c_feas.copy()
            best_radii = r_feas.copy()

    # If still no good result, do a random restart
    if best_sum < 2.0:
        for trial_idx in range(20):
            rng = np.random.RandomState(1000 + trial_idx)
            centers_rand = rng.uniform(0.1, 0.9, (n, 2))
            radii_rand = rng.uniform(0.03, 0.08, n)

            x0 = np.concatenate([centers_rand.flatten(), radii_rand])
            bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

            x_current = x0.copy()
            for pw in [200, 1000, 3000]:
                obj_fn = make_obj_func(n, pw)
                grad_fn = make_grad_func(n, pw)
                try:
                    result = minimize(
                        obj_fn, x_current, method='L-BFGS-B',
                        jac=grad_fn, bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-18}
                    )
                    x_current = result.x
                except Exception:
                    pass

            c_opt = x_current[:2 * n].reshape(n, 2)
            r_opt = x_current[2 * n:]
            c_feas, r_feas = enforce_feasibility(c_opt, r_opt, n)

            sum_r = np.sum(r_feas)
            if sum_r > best_sum:
                best_sum = sum_r
                best_centers = c_feas.copy()
                best_radii = r_feas.copy()

    if best_centers is None:
        best_centers, best_radii = enforce_feasibility(
            x_current[:2 * n].reshape(n, 2),
            x_current[2 * n:], n
        )
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum)
