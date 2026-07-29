# sol_000110 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 82d73ba2) state=fce8fd50 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(x, n, lam):
    """
    Objective function for packing optimization.
    Minimizes -r + penalty for constraint violations.
    """
    centers = x[:n * 2].reshape(n, 2)
    r = x[n * 2]
    if r < 1e-7:
        r = 1e-7
        
    penalty = 0.0
    
    # Boundary constraints: circles must be inside [0,1]x[0,1]
    for i in range(n):
        cx, cy = centers[i]
        # Violation amounts for each boundary
        v = max(0, r - cx, cx - (1 - r), r - cy, cy - (1 - r))
        penalty += v ** 2
        
    # Overlap constraints: distance between centers >= 2r
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            v = max(0, 2 * r - dist)
            penalty += v ** 2
            
    return -r + lam * penalty

def generate_initial_configs(n):
    """
    Generates diverse initial configurations to improve chances of finding global optimum.
    """
    configs = []
    
    # Random initializations with different seeds
    for seed in [42, 123, 456, 789, 1011]:
        np.random.seed(seed)
        configs.append(np.random.rand(n * 2 + 1))
        
    # 5x5 grid plus one in the center
    grid = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < n:
                grid[idx] = [0.1 + i * 0.2, 0.1 + j * 0.2]
                idx += 1
    if idx < n:
        grid[idx] = [0.5, 0.5]
    configs.append(np.concatenate([grid.flatten(), [0.09]]))
    
    # Hexagonal packing pattern
    hex_c = np.zeros((n, 2))
    idx = 0
    row = 0
    while idx < n:
        y = 0.05 + row * 0.1732
        offset = 0.15 if row % 2 == 1 else 0.1
        count = 5 if row % 2 == 0 else 4
        for k in range(count):
            if idx < n:
                hex_c[idx] = [offset + k * 0.2, y]
                idx += 1
        row += 1
    # Add small jitter to break symmetry
    hex_c += np.random.randn(n, 2) * 1e-4
    configs.append(np.concatenate([hex_c.flatten(), [0.1]]))
    
    return configs

def run_packing():
    n = 26
    best_x = None
    best_obj = np.inf
    
    configs = generate_initial_configs(n)
    
    for x0 in configs:
        x0 = x0.copy()
        lam = 100.0
        # Progressively increase penalty weight to enforce constraints while maximizing r
        for _ in range(8):
            res = minimize(compute_objective, x0, args=(n, lam), method='L-BFGS-B',
                           bounds=[(0.0, 1.0)] * (n * 2) + [(0.0, 0.5)],
                           options={'maxiter': 4000, 'ftol': 1e-12, 'gtol': 1e-10})
            x0 = res.x
            lam *= 2.0
            
        r_val = x0[n * 2]
        if -r_val < best_obj:
            best_obj = -r_val
            best_x = x0.copy()
            
    centers = best_x[:n * 2].reshape(n, 2)
    r_final = best_x[n * 2]
    
    # Feasibility correction: shrink radius slightly if any constraint is violated
    max_viol = 0.0
    for i in range(n):
        cx, cy = centers[i]
        v = max(0, r_final - cx, cx - (1 - r_final), r_final - cy, cy - (1 - r_final))
        if v > max_viol:
            max_viol = v
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            v = max(0, 2 * r_final - d) / 2.0
            if v > max_viol:
                max_viol = v
                
    if max_viol > 1e-9:
        r_final -= (max_viol - 1e-12)
        
    radii = np.full(n, r_final)
    return centers, radii, r_final * n
