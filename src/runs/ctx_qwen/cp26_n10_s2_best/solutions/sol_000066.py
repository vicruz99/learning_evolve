# sol_000066 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000026 (state f081a56f) state=ee336c77 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(v):
    """Objective: minimize negative sum of radii."""
    return -v[2*N_CIRCLES:].sum()

def constraint_func(v):
    """
    Computes all inequality constraints:
    1. Boundary: center +/- radius within [0, 1]
    2. Pairwise: distance^2 >= sum_of_radii^2
    Returns a flat array where each element >= 0 indicates a satisfied constraint.
    """
    centers = v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = v[2*N_CIRCLES:]
    
    cons = []
    # Boundary constraints
    cons.append(centers[:, 0] - radii)            # x - r >= 0
    cons.append(1 - centers[:, 0] - radii)        # x + r <= 1
    cons.append(centers[:, 1] - radii)            # y - r >= 0
    cons.append(1 - centers[:, 1] - radii)        # y + r <= 1
    
    # Pairwise non-overlap constraints (vectorized)
    c1 = centers[:, np.newaxis, :]
    c2 = centers[np.newaxis, :, :]
    dist_sq = np.sum((c1 - c2)**2, axis=2)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    cons.append((dist_sq - r_sum_sq)[mask])
    
    return np.concatenate(cons)

def generate_initial_guess(seed):
    """Generates a perturbed hexagonal lattice initialization."""
    np.random.seed(seed)
    r_init = 0.065
    centers = []
    y = r_init
    row = 0
    
    # Generate points in hexagonal pattern
    while len(centers) < N_CIRCLES + 5:
        x_start = r_init if row % 2 == 0 else 2 * r_init
        x = x_start
        while x <= 1 - r_init and len(centers) < N_CIRCLES + 5:
            centers.append([x, y])
            x += 2 * r_init
        y += r_init * np.sqrt(3)
        row += 1
        
    centers = np.array(centers[:N_CIRCLES])
    
    # Add controlled jitter to break symmetry and aid optimization
    centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Start with small feasible radii
    radii = np.full(N_CIRCLES, 0.03)
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    n = N_CIRCLES
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -1.0
    best_x = None
    
    # Multiple restarts with different seeds to escape local minima
    for seed in range(25):
        try:
            x0 = generate_initial_guess(seed)
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
            
            if not np.isnan(res.fun):
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    if best_x is None:
        # Fallback to a valid but suboptimal configuration
        coords = np.linspace(0.1, 0.9, 5)
        centers_fallback = []
        for x in coords:
            for y in coords:
                centers_fallback.append([x, y])
        centers_fallback.append([0.5, 0.05])
        centers = np.array(centers_fallback[:n])
        radii = np.full(n, 0.08)
        return centers, radii, float(np.sum(radii))

    # Extract best result
    centers = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]
    
    # Post-processing to guarantee strict validity per validation rules
    centers = np.clip(centers, 1e-9, 1.0 - 1e-9)
    
    # Enforce boundary constraints strictly
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x - 1e-12, 1.0 - x - 1e-12, y - 1e-12, 1.0 - y - 1e-12)
        
    # Enforce non-overlap strictly with safety margin
    for _ in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    shrink = (sum_r - dist) / 2.0 + 1e-7
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
