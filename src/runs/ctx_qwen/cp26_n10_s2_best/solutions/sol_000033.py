# sol_000033 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000028 (state 1c5b6a86) state=525d2ffd sum of radii=2.607432 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_pair_indices(n):
    """Precompute indices for all unique circle pairs using upper triangle."""
    return np.triu_indices(n, k=1)

def objective(v, n):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraints(v, n, pair_i, pair_j):
    """Compute inequality constraints: boundaries and non-overlap."""
    centers = v[:2*n].reshape(n, 2)
    radii = v[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons = np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    c_i = centers[pair_i]
    c_j = centers[pair_j]
    r_i = radii[pair_i]
    r_j = radii[pair_j]
    
    dist_sq = np.sum((c_i - c_j)**2, axis=1)
    r_sum = r_i + r_j
    
    cons = np.concatenate([cons, dist_sq - r_sum**2])
    return cons

def hex_init(n, scale, jitter):
    """Generate a hexagonal lattice initialization."""
    centers = []
    r_est = 0.08
    y = r_est
    row = 0
    while len(centers) < n:
        x_start = r_est + (row % 2) * r_est
        x = x_start
        while x <= 1.0 - r_est and len(centers) < n:
            centers.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3)
        row += 1
        
    centers = np.array(centers[:n])
    centers = centers * scale + (0.5 - 0.5 * scale)
    centers += np.random.uniform(-jitter, jitter, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    return centers

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    pair_i, pair_j = get_pair_indices(n)
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-4, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Configuration 1: Hexagonal lattice starts with varied scales and jitter
    for s in range(12):
        np.random.seed(s)
        centers = hex_init(n, scale=0.92 + np.random.rand() * 0.08, jitter=0.015)
        radii = np.full(n, 0.06)
        x0 = np.concatenate([centers.flatten(), radii])
        
        res = minimize(objective, x0, args=(n,), method='SLSQP',
                       bounds=bounds,
                       constraints={'type': 'ineq', 'fun': constraints, 'args': (n, pair_i, pair_j)},
                       options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                       
        if res.success:
            centers = res.x[:2*n].reshape(n, 2)
            radii = res.x[2*n:]
            if np.sum(radii) > best_sum:
                best_sum = np.sum(radii)
                best_centers = centers.copy()
                best_radii = radii.copy()
                
    # Configuration 2: Random valid starts
    for s in range(8):
        np.random.seed(100 + s)
        centers = np.random.uniform(0.1, 0.9, size=(n, 2))
        radii = np.full(n, 0.04)
        x0 = np.concatenate([centers.flatten(), radii])
        
        res = minimize(objective, x0, args=(n,), method='SLSQP',
                       bounds=bounds,
                       constraints={'type': 'ineq', 'fun': constraints, 'args': (n, pair_i, pair_j)},
                       options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                       
        if res.success:
            centers = res.x[:2*n].reshape(n, 2)
            radii = res.x[2*n:]
            if np.sum(radii) > best_sum:
                best_sum = np.sum(radii)
                best_centers = centers.copy()
                best_radii = radii.copy()
                
    # Iterative refinement: perturb best solution and re-optimize
    for _ in range(10):
        if best_centers is None:
            break
        x0 = np.concatenate([best_centers.flatten(), best_radii])
        x0 += np.random.uniform(-0.008, 0.008, x0.shape)
        x0[:2*n] = np.clip(x0[:2*n], 0.02, 0.98)
        
        res = minimize(objective, x0, args=(n,), method='SLSQP',
                       bounds=bounds,
                       constraints={'type': 'ineq', 'fun': constraints, 'args': (n, pair_i, pair_j)},
                       options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                       
        if res.success:
            centers = res.x[:2*n].reshape(n, 2)
            radii = res.x[2*n:]
            if np.sum(radii) > best_sum:
                best_sum = np.sum(radii)
                best_centers = centers.copy()
                best_radii = radii.copy()
                
    # Final strict post-processing to guarantee validator compliance
    centers = best_centers
    radii = best_radii
    
    # Enforce boundary constraints strictly
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1.0 - x, y, 1.0 - y)
        
    # Enforce non-overlap strictly with safety margin
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = radii[i] + radii[j]
            if dist < sum_r:
                shrink = (sum_r - dist) / 2.0 + 1e-8
                radii[i] = max(0.0, radii[i] - shrink)
                radii[j] = max(0.0, radii[j] - shrink)
                
    return centers, radii, float(np.sum(radii))
