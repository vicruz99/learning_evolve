# sol_000069 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 16623584) state=9f0e6bd0 sum of radii=2.364967 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize


def compute_distance_matrix(centers):
    n = centers.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            dist[i, j] = d
            dist[j, i] = d
    return dist


def solve_radii_lp(centers, n):
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
    
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(max(0, max_r))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0)
    except Exception:
        pass
    
    return np.full(n, 0.001)


def penalty_objective(x_flat, radii, n):
    centers = x_flat.reshape(-1, 2)
    penalty = 0.0
    weight = 10000.0
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        if x < r:
            penalty += weight * (r - x) ** 2
        if x > 1 - r:
            penalty += weight * (x - (1 - r)) ** 2
        if y < r:
            penalty += weight * (r - y) ** 2
        if y > 1 - r:
            penalty += weight * (y - (1 - r)) ** 2
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += weight * (min_dist - dist) ** 2
    
    return penalty


def gradient_objective(x_flat, radii, n):
    centers = x_flat.reshape(-1, 2)
    grad = np.zeros_like(x_flat)
    weight = 20000.0
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        if x < r:
            grad[2 * i] += weight * (r - x) * (-1)
        if x > 1 - r:
            grad[2 * i] += weight * (x - (1 - r)) * (1)
        if y < r:
            grad[2 * i + 1] += weight * (r - y) * (-1)
        if y > 1 - r:
            grad[2 * i + 1] += weight * (y - (1 - r)) * (1)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist and dist > 1e-12:
                inv_dist = 1.0 / dist
                grad[2 * i] += weight * (min_dist - dist) * dx * inv_dist
                grad[2 * i + 1] += weight * (min_dist - dist) * dy * inv_dist
                grad[2 * j] -= weight * (min_dist - dist) * dx * inv_dist
                grad[2 * j + 1] -= weight * (min_dist - dist) * dy * inv_dist
    
    return grad


def optimize_positions(centers, radii, n):
    x0 = centers.reshape(-1)
    bounds = [(0.0001, 0.9999) for _ in range(2 * n)]
    
    def obj(x):
        return penalty_objective(x, radii, n)
    
    def grad(x):
        return gradient_objective(x, radii, n)
    
    res = minimize(obj, x0, jac=grad, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-10})
    
    return res.x.reshape(-1, 2)


def hexagonal_pattern(n, row_counts, spacing_x, spacing_y_factor=0.866):
    centers = np.zeros((n, 2))
    idx = 0
    
    for row, count in enumerate(row_counts):
        offset = spacing_x / 2 if row % 2 == 1 else 0
        for col in range(count):
            if idx >= n:
                break
            x = 0.08 + col * spacing_x + offset
            y = 0.08 + row * spacing_x * spacing_y_factor
            centers[idx] = [np.clip(x, 0.01, 0.99), np.clip(y, 0.01, 0.99)]
            idx += 1
    
    return centers


def generate_initial_configs(n):
    configs = []
    
    patterns = [
        [5, 4, 5, 4, 5, 3],
        [5, 5, 5, 5, 4, 2],
        [6, 4, 6, 4, 6],
        [4, 6, 4, 6, 4, 2],
        [5, 4, 6, 4, 7],
        [7, 4, 6, 4, 5],
        [6, 6, 6, 6, 2],
        [4, 5, 5, 5, 4, 3],
        [5, 5, 6, 5, 5],
    ]
    
    spacings = [0.16, 0.17, 0.18, 0.19, 0.20]
    
    for pattern in patterns:
        if sum(pattern) != n:
            continue
        for sp in spacings:
            centers = hexagonal_pattern(n, pattern, sp)
            configs.append(centers)
    
    # Random configurations
    for _ in range(15):
        centers = np.random.uniform(0.15, 0.85, (n, 2))
        configs.append(centers)
    
    # Perturb best patterns
    for config in configs[:10]:
        for _ in range(5):
            perturbed = config + np.random.randn(n, 2) * 0.015
            perturbed = np.clip(perturbed, 0.02, 0.98)
            configs.append(perturbed)
    
    return configs


def run_packing():
    np.random.seed(123)
    n = 26
    
    best_centers = None
    best_radii = None
    best_sum = -1
    
    configs = generate_initial_configs(n)
    
    for attempt_idx, init_centers in enumerate(configs):
        centers = init_centers.copy()
        
        prev_sum = -1
        for iteration in range(60):
            radii = solve_radii_lp(centers, n)
            centers = optimize_positions(centers, radii, n)
            
            radii = solve_radii_lp(centers, n)
            current_sum = np.sum(radii)
            
            if abs(current_sum - prev_sum) < 1e-10:
                break
            prev_sum = current_sum
        
        radii = solve_radii_lp(centers, n)
        current_sum = np.sum(radii)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    # Final safety adjustment to ensure validity
    for i in range(n):
        x, y = best_centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        best_radii[i] = min(best_radii[i], max(0, max_r))
    
    # Ensure pairwise non-overlap with small margin
    for i in range(n):
        for j in range(i + 1, n):
            dx = best_centers[i, 0] - best_centers[j, 0]
            dy = best_centers[i, 1] - best_centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = best_radii[i] + best_radii[j]
            if sum_r > dist:
                scale = dist / sum_r
                best_radii[i] *= scale
                best_radii[j] *= scale
    
    # Final re-compute via LP to maximize within constraints
    best_radii = solve_radii_lp(best_centers, n)
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, final_sum
