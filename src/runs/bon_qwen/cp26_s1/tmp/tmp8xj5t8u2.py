import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import functools


def boundary_constraint_x_lower(params, n, i):
    return params[2 * i] - params[2 * n + i]


def boundary_constraint_x_upper(params, n, i):
    return 1.0 - params[2 * i] - params[2 * n + i]


def boundary_constraint_y_lower(params, n, i):
    return params[2 * i + 1] - params[2 * n + i]


def boundary_constraint_y_upper(params, n, i):
    return 1.0 - params[2 * i + 1] - params[2 * n + i]


def overlap_constraint(params, n, i, j):
    dx = params[2 * i] - params[2 * j]
    dy = params[2 * i + 1] - params[2 * j + 1]
    dist_sq = dx * dx + dy * dy
    r_sum = params[2 * n + i] + params[2 * n + j]
    return dist_sq - r_sum * r_sum


def radius_constraint(params, n, i):
    return params[2 * n + i]


def build_constraints(n):
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': functools.partial(boundary_constraint_x_lower, n=n, i=i)})
        constraints.append({'type': 'ineq', 'fun': functools.partial(boundary_constraint_x_upper, n=n, i=i)})
        constraints.append({'type': 'ineq', 'fun': functools.partial(boundary_constraint_y_lower, n=n, i=i)})
        constraints.append({'type': 'ineq', 'fun': functools.partial(boundary_constraint_y_upper, n=n, i=i)})
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': functools.partial(overlap_constraint, n=n, i=i, j=j)})
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': functools.partial(radius_constraint, n=n, i=i)})
    return constraints


def objective_function(params, n):
    radii = params[2 * n:]
    return -np.sum(radii)


def make_initial_hexagonal(n):
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09)
    idx = 0
    row_h = 0.16
    col_w = 0.20
    for row in range(7):
        for col in range(6):
            if idx >= n:
                break
            x = 0.06 + col * col_w + (row % 2) * col_w / 2
            y = 0.06 + row * row_h
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers[idx] = [x, y]
                idx += 1
        if idx >= n:
            break
    for i in range(idx, n):
        centers[i] = [0.5, 0.5]
    return centers, radii


def force_based_optimization(centers, radii, n, max_iters=3000):
    centers = centers.copy()
    radii = radii.copy()
    
    for iteration in range(max_iters):
        forces = np.zeros((n, 2))
        dt = 0.05 / (1 + iteration * 0.002)
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j][0] - centers[i][0]
                dy = centers[j][1] - centers[i][1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-10
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    fx = overlap * dx / dist
                    fy = overlap * dy / dist
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
        
        for i in range(n):
            margin = radii[i]
            if centers[i][0] - margin < 0:
                forces[i][0] += 5 * (margin - centers[i][0])
            if centers[i][0] + margin > 1:
                forces[i][0] -= 5 * (centers[i][0] + margin - 1)
            if centers[i][1] - margin < 0:
                forces[i][1] += 5 * (margin - centers[i][1])
            if centers[i][1] + margin > 1:
                forces[i][1] -= 5 * (centers[i][1] + margin - 1)
        
        centers += forces * dt
        centers = np.clip(centers, 0.001, 0.999)
        
        growth = 0.00005 / (1 + iteration * 0.001)
        for i in range(n):
            max_r = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
            radii[i] = min(radii[i] + growth, max_r)
    
    return centers, radii


def per_circle_optimization(centers, radii, n, max_iters=5000):
    centers = centers.copy()
    radii = radii.copy()
    
    for iteration in range(max_iters):
        forces = np.zeros((n, 2))
        dt = 0.03 / (1 + iteration * 0.001)
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j][0] - centers[i][0]
                dy = centers[j][1] - centers[i][1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-10
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    scale = 0.5
                    fx = scale * overlap * dx / dist
                    fy = scale * overlap * dy / dist
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
        
        for i in range(n):
            margin = radii[i]
            if centers[i][0] - margin < 0:
                forces[i][0] += 3 * (margin - centers[i][0])
            if centers[i][0] + margin > 1:
                forces[i][0] -= 3 * (centers[i][0] + margin - 1)
            if centers[i][1] - margin < 0:
                forces[i][1] += 3 * (margin - centers[i][1])
            if centers[i][1] + margin > 1:
                forces[i][1] -= 3 * (centers[i][1] + margin - 1)
        
        centers += forces * dt
        centers = np.clip(centers, 0.001, 0.999)
        
        for i in range(n):
            max_r = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
            local_max = max_r
            for j in range(n):
                if i != j:
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    local_max = min(local_max, dist - radii[j])
            target = max(0, local_max)
            radii[i] = min(max(radii[i] * 1.0002, radii[i] + 0.00001), target)
    
    return centers, radii


def push_to_boundaries(centers, radii, n, max_iters=2000):
    centers = centers.copy()
    radii = radii.copy()
    
    for iteration in range(max_iters):
        forces = np.zeros((n, 2))
        dt = 0.02 / (1 + iteration * 0.001)
        
        for i in range(n):
            target = np.array([0.0, 0.0])
            best_dist = 0
            for tx, ty in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                d = (centers[i][0] - tx) ** 2 + (centers[i][1] - ty) ** 2
                if d < best_dist or best_dist == 0:
                    best_dist = d
                    target = np.array([tx, ty])
            dir_to_boundary = target - centers[i]
            norm = np.sqrt(np.sum(dir_to_boundary ** 2))
            if norm > 0:
                dir_to_boundary = dir_to_boundary / norm
            forces[i] += dir_to_boundary * 0.5
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j][0] - centers[i][0]
                dy = centers[j][1] - centers[i][1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-10
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    fx = 2 * overlap * dx / dist
                    fy = 2 * overlap * dy / dist
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
        
        for i in range(n):
            margin = radii[i]
            if centers[i][0] - margin < 0:
                forces[i][0] += 10 * (margin - centers[i][0])
            if centers[i][0] + margin > 1:
                forces[i][0] -= 10 * (centers[i][0] + margin - 1)
            if centers[i][1] - margin < 0:
                forces[i][1] += 10 * (margin - centers[i][1])
            if centers[i][1] + margin > 1:
                forces[i][1] -= 10 * (centers[i][1] + margin - 1)
        
        centers += forces * dt
        centers = np.clip(centers, 0.001, 0.999)
    
    return centers, radii


def scipy_optimize(centers, radii, n):
    x0 = np.concatenate([centers.flatten(), radii])
    
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
    
    constraints = build_constraints(n)
    
    obj_func = functools.partial(objective_function, n=n)
    
    result = minimize(obj_func, x0, method='SLSQP', bounds=bounds,
                      constraints=constraints,
                      options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
    
    centers_opt = result.x[:2 * n].reshape(n, 2)
    radii_opt = result.x[2 * n:]
    
    return centers_opt, radii_opt


def post_process(centers, radii, n):
    centers = centers.copy()
    radii = radii.copy()
    
    for _ in range(500):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                r_sum = radii[i] + radii[j]
                if dist < r_sum - 1e-12:
                    scale = (dist + 1e-12) / r_sum
                    new_sum = scale * r_sum
                    radii[i] = new_sum * radii[i] / r_sum
                    radii[j] = new_sum * radii[j] / r_sum
                    changed = True
        
        for i in range(n):
            max_r = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
            if radii[i] > max_r + 1e-12:
                radii[i] = max(max_r, 0)
                changed = True
        
        if not changed:
            break
    
    radii = np.maximum(radii, 0)
    return centers, radii


def run_packing():
    n = 26
    
    centers, radii = make_initial_hexagonal(n)
    
    centers, radii = force_based_optimization(centers, radii, n, max_iters=5000)
    
    centers, radii = per_circle_optimization(centers, radii, n, max_iters=5000)
    
    centers, radii = scipy_optimize(centers, radii, n)
    centers, radii = post_process(centers, radii, n)
    
    centers, radii = push_to_boundaries(centers, radii, n, max_iters=3000)
    
    for i in range(n):
        max_r = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
        for j in range(n):
            if i != j:
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, dist - radii[j])
        radii[i] = max(0, max_r)
    
    centers, radii = scipy_optimize(centers, radii, n)
    centers, radii = post_process(centers, radii, n)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii