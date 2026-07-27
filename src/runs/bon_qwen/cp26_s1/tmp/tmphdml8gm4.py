import numpy as np
from scipy.optimize import minimize
import math

N_CIRCLES = 26


def objective(params):
    radii = params[2::3]
    return -np.sum(radii)


def constraints_func(params):
    n = N_CIRCLES
    c = []
    
    for i in range(n):
        c.append(params[3 * i] - params[3 * i + 2])
        c.append(1.0 - params[3 * i] - params[3 * i + 2])
        c.append(params[3 * i + 1] - params[3 * i + 2])
        c.append(1.0 - params[3 * i + 1] - params[3 * i + 2])
        c.append(params[3 * i + 2] - 0.0001)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = params[3 * i] - params[3 * j]
            dy = params[3 * i + 1] - params[3 * j + 1]
            dr = params[3 * i + 2] + params[3 * j + 2]
            c.append(dx * dx + dy * dy - dr * dr)
    
    return np.array(c)


def initialize_hexagonal(n):
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.095
    
    idx = 0
    row = 0
    while idx < n:
        if row % 2 == 0:
            n_in_row = 6
        else:
            n_in_row = 5
        
        if idx + n_in_row > n:
            n_in_row = n - idx
        
        y = (row + 1) * (1.0 / 7)
        x_spacing = 1.0 / (n_in_row + 1)
        
        for col in range(n_in_row):
            if idx >= n:
                break
            x = (col + 1) * x_spacing
            if row % 2 == 1:
                x += x_spacing * 0.5
            centers[idx] = [x, y]
            idx += 1
        row += 1
    
    return centers, radii


def initialize_random(n, seed):
    np.random.seed(seed)
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.08
    
    for i in range(n):
        centers[i, 0] = np.random.uniform(0.1, 0.9)
        centers[i, 1] = np.random.uniform(0.1, 0.9)
    
    return centers, radii


def force_optimize(centers, radii, n, iterations=15000):
    centers = centers.copy()
    radii = radii.copy()
    
    lr = 0.015
    
    for iteration in range(iterations):
        expansion = 1.0 + 0.00008 * (1.0 - iteration / iterations)
        radii *= expansion
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist_sq = dx * dx + dy * dy
                dist = math.sqrt(dist_sq)
                
                sum_r = radii[i] + radii[j]
                
                if dist < sum_r and dist > 1e-10:
                    overlap = sum_r - dist
                    force = min(overlap / dist, 0.5) * lr
                    centers[i, 0] -= force * dx
                    centers[i, 1] -= force * dy
                    centers[j, 0] += force * dx
                    centers[j, 1] += force * dy
        
        for i in range(n):
            r = radii[i]
            centers[i, 0] = max(r, min(1.0 - r, centers[i, 0]))
            centers[i, 1] = max(r, min(1.0 - r, centers[i, 1]))
        
        if iteration % 3000 == 0 and iteration > 0:
            lr *= 0.7
    
    return centers, radii


def optimize_single(centers_init, radii_init, n):
    centers, radii = force_optimize(centers_init, radii_init, n, iterations=12000)
    
    params = np.zeros(3 * n)
    for i in range(n):
        params[3 * i] = centers[i, 0]
        params[3 * i + 1] = centers[i, 1]
        params[3 * i + 2] = radii[i]
    
    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.5))
        bounds.append((0.001, 0.5))
        bounds.append((0.001, 0.5))
    
    constraint = {'type': 'ineq', 'fun': constraints_func}
    
    try:
        result = minimize(
            objective,
            params,
            method='SLSQP',
            constraints=constraint,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < -1.0:
            centers_opt = result.x.reshape((n, 3))[:, :2]
            radii_opt = result.x.reshape((n, 3))[:, 2]
            return centers_opt, radii_opt
    except Exception:
        pass
    
    return centers, radii


def run_packing():
    n = N_CIRCLES
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    centers_init, radii_init = initialize_hexagonal(n)
    centers_opt, radii_opt = optimize_single(centers_init, radii_init, n)
    
    total = np.sum(radii_opt)
    if total > best_sum:
        best_sum = total
        best_centers = centers_opt.copy()
        best_radii = radii_opt.copy()
    
    for seed in range(30):
        centers_init, radii_init = initialize_hexagonal(n)
        centers_init += np.random.RandomState(seed).rand(n, 2) * 0.04
        radii_init += np.random.RandomState(seed).rand(n) * 0.01
        
        centers_opt, radii_opt = optimize_single(centers_init, radii_init, n)
        
        total = np.sum(radii_opt)
        if total > best_sum:
            best_sum = total
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()
    
    for seed in range(15):
        centers_init, radii_init = initialize_random(n, seed)
        centers_opt, radii_opt = optimize_single(centers_init, radii_init, n)
        
        total = np.sum(radii_opt)
        if total > best_sum:
            best_sum = total
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()
    
    # Final force-based refinement on best solution
    best_centers, best_radii = force_optimize(best_centers, best_radii, n, iterations=8000)
    
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, best_sum