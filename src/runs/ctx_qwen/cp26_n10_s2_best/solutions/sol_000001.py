# sol_000001 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 77dfa116) state=1501c8b5 sum of radii=2.626292 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(v):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -v[2*N_CIRCLES:].sum()

def constraint_func(v):
    """
    Computes all inequality constraints:
    1. Boundary: center +/- radius within [0, 1]
    2. Pairwise: distance >= sum of radii
    Returns a flat array where each element >= 0 indicates a satisfied constraint.
    """
    centers = v[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = v[2*N_CIRCLES:]
    
    constr = []
    # Boundary constraints
    constr.append(centers[:, 0] - radii)            # x - r >= 0
    constr.append(1 - centers[:, 0] - radii)        # x + r <= 1
    constr.append(centers[:, 1] - radii)            # y - r >= 0
    constr.append(1 - centers[:, 1] - radii)        # y + r <= 1
    
    # Pairwise non-overlap constraints (vectorized upper triangle)
    c1 = centers[:, np.newaxis, :]
    c2 = centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum((c1 - c2)**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    mask = np.triu(np.ones((N_CIRCLES, N_CIRCLES), dtype=bool), k=1)
    constr.append((dists - r_sum)[mask])
    
    return np.concatenate(constr)

def run_packing():
    n = N_CIRCLES
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Multiple restarts with different seeds to escape local minima
    for seed in range(8):
        np.random.seed(seed)
        
        # Structured grid initialization
        init_centers = np.zeros((n, 2))
        init_radii = np.full(n, 0.035)  # Start small to guarantee feasibility
        count = 0
        for r in range(6):
            for c in range(5):
                if count < n:
                    x = 0.1 + c * 0.17
                    y = 0.1 + r * 0.17
                    init_centers[count] = [x, y]
                    count += 1
                    
        # Shuffle to break symmetry
        perm = np.random.permutation(n)
        init_centers = init_centers[perm]
        
        # Add controlled jitter
        jitter = np.random.uniform(-0.01, 0.01, size=init_centers.shape)
        init_centers = np.clip(init_centers + jitter, 0.05, 0.95)
        
        init_vars = np.concatenate([init_centers.flatten(), init_radii])
        
        # Optimize using SLSQP
        res = minimize(objective_func, init_vars, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': constraint_func},
                       options={'maxiter': 2500, 'ftol': 1e-10, 'disp': False})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = res.x[:2*n].reshape(n, 2).copy()
            best_radii = res.x[2*n:].copy()
            
    # Post-processing to guarantee strict validity per validation rules
    centers = best_centers
    radii = best_radii
    
    # Enforce boundary constraints strictly
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1-x, y, 1-y)
        
    # Enforce non-overlap strictly with safety margin
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            sum_r = radii[i] + radii[j]
            if dist < sum_r:
                shrink = (sum_r - dist) / 2.0 + 1e-7
                radii[i] = max(0.0, radii[i] - shrink)
                radii[j] = max(0.0, radii[j] - shrink)
                
    return centers, radii, np.sum(radii)
