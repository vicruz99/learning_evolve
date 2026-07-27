# sol_000083 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9821b492) state=15729d36 sum of radii=2.400000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def compute_optimal_radii(centers):
    n = centers.shape[0]
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        x, y = centers[i]
        row = np.zeros(n)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(x)
        A_ub.append(row.copy())
        b_ub.append(1 - x)
        A_ub.append(row.copy())
        b_ub.append(y)
        A_ub.append(row.copy())
        b_ub.append(1 - y)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * n
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        return result.x
    else:
        return np.full(n, 0.05)

def position_penalty(params, radii, n=26):
    centers = params.reshape(n, 2)
    penalty = 0.0
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        for coord in [x, y]:
            if coord < r:
                penalty += 100.0 * (r - coord) ** 2
            if coord > 1.0 - r:
                penalty += 100.0 * (coord - (1.0 - r)) ** 2
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += 200.0 * (min_dist - dist) ** 2
    
    return penalty

def optimize_positions(centers, radii, n=26):
    best_centers = centers.copy()
    best_penalty = position_penalty(centers.flatten(), radii, n)
    
    for _ in range(5):
        result = minimize(
            position_penalty,
            best_centers.flatten(),
            args=(radii, n),
            method='L-BFGS-B',
            bounds=[(0.01, 0.99)] * (2 * n),
            options={'maxiter': 5000, 'ftol': 1e-12}
        )
        if result.fun < best_penalty:
            best_centers = result.x.reshape(n, 2)
            best_penalty = result.fun
    
    return best_centers

def generate_hexagonal_packing(n=26, rows_config=None):
    if rows_config is None:
        rows_config = [6, 5, 6, 5, 4]
    
    centers = np.zeros((n, 2))
    idx = 0
    n_rows = len(rows_config)
    
    for row_idx, num_circles in enumerate(rows_config):
        y = (row_idx + 0.5) / n_rows
        
        if row_idx % 2 == 0:
            x_positions = np.linspace(0.5 / num_circles, 1.0 - 0.5 / num_circles, num_circles)
        else:
            x_positions = np.linspace(1.0 / (2.0 * num_circles), 1.0 - 1.0 / (2.0 * num_circles), num_circles)
        
        for col_idx in range(num_circles):
            centers[idx] = [x_positions[col_idx], y]
            idx += 1
    
    return centers

def run_packing():
    n = 26
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    row_configs = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [6, 6, 5, 5, 4],
        [5, 5, 6, 6, 4],
        [6, 5, 5, 6, 4],
        [4, 6, 6, 5, 5],
    ]
    
    for rows_config in row_configs:
        if sum(rows_config) != n:
            continue
        
        centers = generate_hexagonal_packing(n, rows_config)
        
        for _ in range(3):
            radii = compute_optimal_radii(centers)
            centers = optimize_positions(centers, radii, n)
        
        radii = compute_optimal_radii(centers)
        s = np.sum(radii)
        
        if s > best_sum:
            best_sum = s
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    for _ in range(5):
        perturbed = best_centers + np.random.randn(n, 2) * 0.03
        perturbed = np.clip(perturbed, 0.05, 0.95)
        
        radii = compute_optimal_radii(perturbed)
        
        for _ in range(2):
            centers_temp = optimize_positions(perturbed, radii, n)
            radii = compute_optimal_radii(centers_temp)
        
        s = np.sum(radii)
        if s > best_sum:
            best_sum = s
            best_centers = centers_temp.copy()
            best_radii = radii.copy()
    
    for _ in range(10):
        radii = compute_optimal_radii(best_centers)
        best_centers = optimize_positions(best_centers, radii, n)
        radii = compute_optimal_radii(best_centers)
    
    best_radii = compute_optimal_radii(best_centers)
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, best_sum
