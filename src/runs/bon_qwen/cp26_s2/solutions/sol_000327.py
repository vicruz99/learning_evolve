# sol_000327 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cc22fbce) state=7ac3db57 sum of radii=2.031412 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog


def compute_max_radii_lp(centers):
    """Solve LP to maximize sum of radii given fixed center positions."""
    n = centers.shape[0]
    
    c_obj = -np.ones(n)
    
    constraints = []
    rhs = []
    
    for i in range(n):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0],
                     centers[i, 1], 1.0 - centers[i, 1])
        row = np.zeros(n)
        row[i] = 1.0
        constraints.append(row)
        rhs.append(max_r)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            constraints.append(row)
            rhs.append(dist)
    
    A_ub = np.array(constraints)
    b_ub = np.array(rhs)
    
    bounds = [(0.0, None) for _ in range(n)]
    
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        return result.x
    else:
        return np.ones(n) * 0.01


def relax_positions_force(centers, radii, lr=0.005, n_iters=800):
    """Force-based relaxation to resolve overlaps and boundary violations."""
    n = centers.shape[0]
    
    for it in range(n_iters):
        forces = np.zeros_like(centers)
        current_lr = lr * max(0.01, 1.0 - it / n_iters)
        
        for i in range(n):
            for d in range(2):
                if centers[i, d] < radii[i]:
                    forces[i, d] += 200.0 * (radii[i] - centers[i, d])
                if centers[i, d] > 1.0 - radii[i]:
                    forces[i, d] -= 200.0 * (centers[i, d] - (1.0 - radii[i]))
            
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx * dx + dy * dy
                dist = np.sqrt(dist_sq) if dist_sq > 1e-20 else 1e-10
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    repel = 80.0 * overlap / dist
                    forces[i, 0] += dx * repel
                    forces[i, 1] += dy * repel
                    forces[j, 0] -= dx * repel
                    forces[j, 1] -= dy * repel
        
        centers = centers + current_lr * forces
        
        for i in range(n):
            centers[i, 0] = max(radii[i], min(1.0 - radii[i], centers[i, 0]))
            centers[i, 1] = max(radii[i], min(1.0 - radii[i], centers[i, 1]))
    
    return centers


def run_packing():
    np.random.seed(42)
    n = 26
    
    centers = np.zeros((n, 2))
    idx = 0
    
    for row in range(6):
        count = 5 if row < 5 else 1
        y = 0.06 + row * 0.15
        offset = 0.075 if row % 2 == 1 else 0.0
        for col in range(count):
            x = 0.06 + col * 0.17 + offset
            if idx < n:
                centers[idx, 0] = x
                centers[idx, 1] = y
                idx += 1
    
    radii = np.ones(n) * 0.04
    
    best_sum = 0.0
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    for outer in range(80):
        new_radii = compute_max_radii_lp(centers)
        if np.any(np.isnan(new_radii)):
            new_radii = radii
        
        new_radii = np.maximum(new_radii, 0.001)
        
        centers = relax_positions_force(centers, new_radii, lr=0.006, n_iters=600)
        radii = new_radii
        
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
    
    final_radii = compute_max_radii_lp(best_centers)
    if np.any(np.isnan(final_radii)):
        final_radii = best_radii
    else:
        final_radii = np.maximum(final_radii, 0.001)
    
    best_centers = relax_positions_force(best_centers, final_radii, lr=0.004, n_iters=1000)
    
    return best_centers, final_radii, np.sum(final_radii)
