# sol_000166 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3abb93ac) state=78a7ac00 sum of radii=2.400000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog


def position_penalty(x, n, radii):
    centers = x.reshape(n, 2)
    p = 0.0
    for i in range(n):
        xi = centers[i, 0]
        yi = centers[i, 1]
        ri = radii[i]
        if xi < ri:
            p += 1000.0 * (xi - ri) ** 2
        if xi > 1.0 - ri:
            p += 1000.0 * (xi - (1.0 - ri)) ** 2
        if yi < ri:
            p += 1000.0 * (yi - ri) ** 2
        if yi > 1.0 - ri:
            p += 1000.0 * (yi - (1.0 - ri)) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                p += 1000.0 * (min_dist - dist) ** 2
    return p


def position_penalty_grad(x, n, radii):
    centers = x.reshape(n, 2)
    grad = np.zeros_like(x)
    for i in range(n):
        xi = centers[i, 0]
        yi = centers[i, 1]
        ri = radii[i]
        if xi < ri:
            grad[2 * i] += 2000.0 * (xi - ri)
        if xi > 1.0 - ri:
            grad[2 * i] += 2000.0 * (xi - (1.0 - ri))
        if yi < ri:
            grad[2 * i + 1] += 2000.0 * (yi - ri)
        if yi > 1.0 - ri:
            grad[2 * i + 1] += 2000.0 * (yi - (1.0 - ri))
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist and dist > 1e-10:
                overlap = min_dist - dist
                fx = 2000.0 * overlap * (dx / dist)
                fy = 2000.0 * overlap * (dy / dist)
                grad[2 * i] += fx
                grad[2 * i + 1] += fy
                grad[2 * j] -= fx
                grad[2 * j + 1] -= fy
    return grad


def optimize_radii_lp(centers):
    n = len(centers)
    c = -np.ones(n)
    A_ub_rows = []
    b_ub_vals = []
    for i in range(n):
        x = centers[i, 0]
        y = centers[i, 1]
        for val in [x, 1.0 - x, y, 1.0 - y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(val)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(dist)
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    bounds = [(0.0, None)] * n
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if result.success:
        return result.x
    return np.zeros(n)


def optimize_positions_penalty(centers, radii, maxiter=2000):
    n = len(centers)
    x0 = centers.flatten()
    bounds = [(0.0, 1.0)] * (2 * n)

    def obj(x):
        return position_penalty(x, n, radii)

    def grad(x):
        return position_penalty_grad(x, n, radii)

    result = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                      bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10})
    return result.x.reshape(n, 2)


def make_hexagonal_init(n, rows, r):
    centers_list = []
    y = r
    for row_idx, num in enumerate(rows):
        if row_idx % 2 == 1:
            x_start = 2.0 * r
        else:
            x_start = r
        for col in range(num):
            x = x_start + col * 2.0 * r
            centers_list.append([x, y])
        y += np.sqrt(3.0) * r
    centers = np.array(centers_list[:n])
    radii = np.full(n, r)
    return centers, radii


def make_grid_init(n, grid_size, r):
    centers_list = []
    y = r
    count = 0
    while count < n:
        x = r
        while count < n:
            centers_list.append([x, y])
            count += 1
            x += 2.0 * r
            if x + r > 1.0:
                break
        y += 2.0 * r
        if y + r > 1.0 and count < n:
            while count < n:
                centers_list.append([0.5, 0.5])
                count += 1
            break
    centers = np.array(centers_list[:n])
    radii = np.full(n, r)
    return centers, radii


def make_random_init(n, r):
    centers = np.random.uniform(r, 1.0 - r, (n, 2))
    radii = np.full(n, r)
    return centers, radii


def alternating_optimize(centers, radii, max_iterations=80):
    n = 26
    best_sum = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()

    for iteration in range(max_iterations):
        # Optimize radii using LP
        radii = optimize_radii_lp(centers)
        if np.sum(radii) <= 0:
            break

        # Optimize positions using penalty method
        centers = optimize_positions_penalty(centers, radii, maxiter=1500)

        current_sum = np.sum(radii)
        # Re-optimize radii after position change
        radii = optimize_radii_lp(centers)
        current_sum = np.sum(radii)

        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

        if iteration > 10 and current_sum < best_sum - 1e-8:
            pass  # Continue trying

    return best_centers, best_radii, best_sum


def refine_with_slsqp(centers, radii):
    n = 26

    def objective(x):
        return -np.sum(x[52:])

    def constraints_vec(x):
        n_loc = 26
        con = []
        for i in range(n_loc):
            con.append(x[2 * i] - x[52 + i])
            con.append(1.0 - x[2 * i] - x[52 + i])
            con.append(x[2 * i + 1] - x[52 + i])
            con.append(1.0 - x[2 * i + 1] - x[52 + i])
        for i in range(n_loc):
            for j in range(i + 1, n_loc):
                dx = x[2 * i] - x[2 * j]
                dy = x[2 * i + 1] - x[2 * j + 1]
                dist = np.sqrt(dx * dx + dy * dy)
                con.append(dist - x[52 + i] - x[52 + j])
        return np.array(con)

    x0 = np.concatenate([centers.flatten(), radii])
    cons = {'type': 'ineq', 'fun': constraints_vec}
    bounds = [(0.0, 1.0)] * 52 + [(1e-8, 0.5)] * 26

    result = minimize(objective, x0, method='SLSQP', constraints=[cons],
                      bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})

    centers_opt = result.x[:52].reshape(26, 2)
    radii_opt = result.x[52:]
    return centers_opt, radii_opt, np.sum(radii_opt)


def run_packing():
    np.random.seed(42)
    n = 26

    best_centers = None
    best_radii = None
    best_sum = 0.0

    # Try multiple initial configurations
    inits = []

    # Hexagonal patterns with different row distributions
    hex_configs = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [4, 6, 5, 6, 5],
        [5, 5, 5, 5, 6],
        [6, 5, 5, 5, 5],
        [5, 6, 6, 5, 4],
        [4, 5, 6, 6, 5],
        [6, 6, 5, 5, 4],
        [5, 5, 6, 6, 4],
        [6, 6, 6, 4, 4],
        [4, 6, 6, 6, 4],
        [6, 4, 6, 6, 4],
    ]

    for rows in hex_configs:
        for r in [0.08, 0.085, 0.09, 0.095, 0.1]:
            centers, radii = make_hexagonal_init(n, rows, r)
            # Check if initialization is valid
            if len(centers) == n and np.all(centers[:, 0] + radii <= 1.0 + 1e-6) and np.all(centers[:, 1] + radii <= 1.0 + 1e-6):
                inits.append((centers.copy(), radii.copy()))

    # Grid initializations
    for r in [0.08, 0.09, 0.1]:
        centers, radii = make_grid_init(n, 5, r)
        if len(centers) == n:
            inits.append((centers.copy(), radii.copy()))

    # Random initializations
    for seed in range(10):
        np.random.seed(seed)
        r = 0.08
        centers = np.random.uniform(r, 1.0 - r, (n, 2))
        radii = np.full(n, r)
        inits.append((centers.copy(), radii.copy()))

    for idx, (centers, radii) in enumerate(inits):
        current_centers, current_radii, current_sum = alternating_optimize(
            centers, radii, max_iterations=60)

        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = current_centers.copy()
            best_radii = current_radii.copy()

    # Final refinement with SLSQP on the best solution
    if best_centers is not None:
        final_centers, final_radii, final_sum = refine_with_slsqp(best_centers, best_radii)
        if final_sum > best_sum:
            best_sum = final_sum
            best_centers = final_centers.copy()
            best_radii = final_radii.copy()

    # Second round of alternating optimization from the SLSQP result
    if best_centers is not None:
        second_centers, second_radii, second_sum = alternating_optimize(
            best_centers, best_radii, max_iterations=60)
        if second_sum > best_sum:
            best_sum = second_sum
            best_centers = second_centers.copy()
            best_radii = second_radii.copy()

        # Third SLSQP refinement
        third_centers, third_radii, third_sum = refine_with_slsqp(best_centers, best_radii)
        if third_sum > best_sum:
            best_sum = third_sum
            best_centers = third_centers.copy()
            best_radii = third_radii.copy()

    # Final LP to ensure radii are optimal for the final positions
    best_radii = optimize_radii_lp(best_centers)
    best_sum = np.sum(best_radii)

    # Validate and clean up
    best_centers = np.clip(best_centers, 0.0, 1.0)
    best_radii = np.maximum(best_radii, 0.0)

    return best_centers, best_radii, best_sum
