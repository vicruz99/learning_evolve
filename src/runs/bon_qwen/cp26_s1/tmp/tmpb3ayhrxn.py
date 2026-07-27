import numpy as np
from scipy.optimize import linprog, minimize

def initial_packing(n_circles=26):
    centers = np.zeros((n_circles, 2))
    radii = np.ones(n_circles) * 0.06
    idx = 0
    row = 0
    while idx < n_circles:
        n_in_row = 6 if row % 2 == 0 else 5
        for col in range(n_in_row):
            if idx >= n_circles:
                break
            x = 0.08 + col * 0.16
            y = 0.08 + row * 0.17
            centers[idx] = [x, y]
            radii[idx] = 0.06
            idx += 1
        row += 1
    return centers, radii


def spread_circles(centers, radii, n_circles, n_iterations=500):
    centers = centers.copy()
    for it in range(n_iterations):
        forces = np.zeros_like(centers)
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    diff[0] += 1e-8
                    dist = 1e-8
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    force_mag = (min_dist - dist) / dist
                    forces[i] += force_mag * diff
                    forces[j] -= force_mag * diff
        for i in range(n_circles):
            r = radii[i]
            if centers[i, 0] - r < 0:
                forces[i, 0] += (r - centers[i, 0]) * 15
            if centers[i, 0] + r > 1:
                forces[i, 0] -= (centers[i, 0] + r - 1) * 15
            if centers[i, 1] - r < 0:
                forces[i, 1] += (r - centers[i, 1]) * 15
            if centers[i, 1] + r > 1:
                forces[i, 1] -= (centers[i, 1] + r - 1) * 15
        lr = 0.03 * np.exp(-it / 150)
        centers += lr * forces
        for i in range(n_circles):
            r = radii[i]
            centers[i, 0] = max(r + 1e-9, min(1 - r - 1e-9, centers[i, 0]))
            centers[i, 1] = max(r + 1e-9, min(1 - r - 1e-9, centers[i, 1]))
    return centers


def optimize_radii(centers, n_circles):
    c = -np.ones(n_circles)
    A_ub = []
    b_ub = []
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n_circles)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
    if len(A_ub) == 0:
        return np.ones(n_circles) * 0.5
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = []
    for i in range(n_circles):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        bounds.append((0, max(1e-12, max_r)))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x
    return np.full(n_circles, 0.05)


def smooth_objective(vars, n_circles):
    centers = vars[:2 * n_circles].reshape(n_circles, 2)
    radii = vars[2 * n_circles:]
    obj = -np.sum(radii)
    for i in range(n_circles):
        r = radii[i]
        obj += 10000 * max(0, r - centers[i, 0]) ** 2
        obj += 10000 * max(0, centers[i, 0] + r - 1) ** 2
        obj += 10000 * max(0, r - centers[i, 1]) ** 2
        obj += 10000 * max(0, centers[i, 1] + r - 1) ** 2
        obj += 10000 * max(0, -r) ** 2
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.linalg.norm(centers[i] - centers[j])
            overlap = radii[i] + radii[j] - dist
            obj += 10000 * max(0, overlap) ** 2
    return obj


def fine_tune(centers, radii, n_circles):
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = []
    for i in range(n_circles):
        x, y = centers[i]
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    result = minimize(smooth_objective, x0, args=(n_circles,), method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15})
    centers_opt = result.x[:2 * n_circles].reshape(n_circles, 2)
    radii_opt = result.x[2 * n_circles:]
    radii_opt = np.maximum(radii_opt, 1e-10)
    for i in range(n_circles):
        r = radii_opt[i]
        centers_opt[i, 0] = max(r + 1e-9, min(1 - r - 1e-9, centers_opt[i, 0]))
        centers_opt[i, 1] = max(r + 1e-9, min(1 - r - 1e-9, centers_opt[i, 1]))
    return centers_opt, radii_opt


def run_packing():
    n_circles = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0

    for trial in range(20):
        np.random.seed(trial * 7 + 3)
        centers, radii = initial_packing(n_circles)
        centers += np.random.uniform(-0.012, 0.012, centers.shape)

        for _ in range(12):
            centers = spread_circles(centers, radii, n_circles, 300)
            radii = optimize_radii(centers, n_circles)

        centers, radii = fine_tune(centers, radii, n_circles)
        radii = optimize_radii(centers, n_circles)
        centers = spread_circles(centers, radii, n_circles, 200)
        radii = optimize_radii(centers, n_circles)

        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    for i in range(n_circles):
        r = best_radii[i]
        best_centers[i, 0] = max(r + 1e-9, min(1 - r - 1e-9, best_centers[i, 0]))
        best_centers[i, 1] = max(r + 1e-9, min(1 - r - 1e-9, best_centers[i, 1]))

    return best_centers, best_radii, np.sum(best_radii)