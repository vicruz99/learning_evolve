import numpy as np
from scipy.optimize import minimize


def run_packing():
    n = 26
    
    best_sum = 0
    best_centers = None
    best_radii = None
    
    for trial in range(40):
        centers, radii = initialize_packing(n, trial)
        
        # Phase 1: Constrained optimization with SLSQP
        centers, radii = optimize_slsqp(centers, radii, n, max_iter=3000)
        
        if is_valid(centers, radii, n):
            # Phase 2: Penalty-based optimization to escape local minima
            centers, radii = optimize_penalty(centers, radii, n, max_iter=2000)
            # Phase 3: Refine with SLSQP again
            centers, radii = optimize_slsqp(centers, radii, n, max_iter=2000)
        
        if is_valid(centers, radii, n):
            current_sum = np.sum(radii)
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
    
    # Fallback if no valid configuration found
    if best_centers is None:
        best_centers, best_radii = initialize_packing(n, 0)
        best_radii = np.ones(n) * 0.02
        best_sum = np.sum(best_radii)
    
    if best_radii is not None:
        best_radii = np.maximum(best_radii, 1e-10)
    
    return best_centers, best_radii, np.sum(best_radii)


def initialize_packing(n, trial):
    np.random.seed(trial * 17 + 3)
    
    hex_configs = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [7, 5, 6, 5, 3],
        [6, 6, 5, 6, 3],
        [5, 5, 6, 6, 4],
        [7, 6, 5, 6, 2],
        [6, 5, 7, 5, 3],
        [4, 6, 6, 6, 4],
        [5, 6, 6, 5, 4],
        [6, 7, 5, 6, 2],
    ]
    
    if trial < 30:
        config = hex_configs[trial % len(hex_configs)]
        centers = hexagonal_config(config, n, trial)
    else:
        centers = grid_config(n, trial)
    
    radii = np.ones(n) * 0.025
    return centers, radii


def hexagonal_config(config, n, trial):
    centers = []
    num_rows = len(config)
    
    for row_idx, count in enumerate(config):
        y = (row_idx + 1) * (1.0 / (num_rows + 1))
        x_spacing = 1.0 / (count + 1)
        
        for col in range(count):
            x = (col + 1) * x_spacing
            centers.append([x, y])
    
    centers = np.array(centers[:n])
    centers += np.random.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    return centers


def grid_config(n, trial):
    centers = []
    count = 0
    
    for i in range(5):
        for j in range(5):
            if count >= n:
                break
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            centers.append([x, y])
            count += 1
        if count >= n:
            break
    
    while len(centers) < n:
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        centers.append([x, y])
    
    centers = np.array(centers[:n])
    centers += np.random.normal(0, 0.01, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    return centers


def is_valid(centers, radii, n):
    if np.any(np.isnan(centers)) or np.any(np.isnan(radii)):
        return False
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-8 or x + r > 1 + 1e-8:
            return False
        if y - r < -1e-8 or y + r > 1 + 1e-8:
            return False
        if r < -1e-8:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - 1e-8:
                return False
    
    return True


def optimize_slsqp(centers, radii, n, max_iter=3000):
    vars_init = np.concatenate([centers.flatten(), radii])
    
    cons = {'type': 'ineq', 'fun': _slsqp_constraints_fn, 'args': (n,)}
    
    result = minimize(_slsqp_objective, vars_init, args=(n,), 
                     constraints=cons, method='SLSQP',
                     options={'maxiter': max_iter, 'ftol': 1e-15, 'disp': False})
    
    x_opt = result.x[:n]
    y_opt = result.x[n:2 * n]
    r_opt = result.x[2 * n:]
    
    return np.column_stack([x_opt, y_opt]), r_opt


def _slsqp_objective(vars, n):
    r = vars[2 * n:]
    return -np.sum(r)


def _slsqp_constraints_fn(vars, n):
    x = vars[:n]
    y = vars[n:2 * n]
    r = vars[2 * n:3 * n]
    
    c = []
    c.extend(x - r)
    c.extend(1 - x - r)
    c.extend(y - r)
    c.extend(1 - y - r)
    c.extend(r)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx * dx + dy * dy
            sum_r = r[i] + r[j]
            c.append(dist_sq - sum_r * sum_r)
    
    return np.array(c)


def optimize_penalty(centers, radii, n, max_iter=2000, penalty=10000):
    vars_init = np.concatenate([centers.flatten(), radii])
    
    result = minimize(_penalty_objective, vars_init, args=(n, penalty), method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})
    
    x_opt = result.x[:n]
    y_opt = result.x[n:2 * n]
    r_opt = result.x[2 * n:]
    
    return np.column_stack([x_opt, y_opt]), r_opt


def _penalty_objective(vars, n, penalty):
    x = vars[:n]
    y = vars[n:2 * n]
    r = vars[2 * n:3 * n]
    
    obj = -np.sum(r)
    pen = 0
    
    for i in range(n):
        if x[i] < r[i]:
            pen += penalty * (r[i] - x[i]) ** 2
        if x[i] > 1 - r[i]:
            pen += penalty * (x[i] - (1 - r[i])) ** 2
        if y[i] < r[i]:
            pen += penalty * (r[i] - y[i]) ** 2
        if y[i] > 1 - r[i]:
            pen += penalty * (y[i] - (1 - r[i])) ** 2
        if r[i] < 0:
            pen += penalty * r[i] ** 2
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = r[i] + r[j]
            if dist < sum_r:
                pen += penalty * (sum_r - dist) ** 2
    
    return obj + pen