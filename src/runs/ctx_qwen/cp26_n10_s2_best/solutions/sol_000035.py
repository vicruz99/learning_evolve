# sol_000035 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000028 (state 1c5b6a86) state=6ded54d8 sum of radii=2.621248 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_function(vars_, n):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(vars_[2*n:])

def constraint_function(vars_, n, pair_i, pair_j):
    """
    Computes inequality constraints:
    1. Boundary: center +/- radius within [0, 1]
    2. Pairwise: squared distance >= squared sum of radii
    Returns a flat array where each element >= 0 indicates a satisfied constraint.
    """
    centers = vars_[:2*n].reshape(n, 2)
    radii = vars_[2*n:]
    
    # Boundary constraints
    c1 = centers[:, 0] - radii
    c2 = 1.0 - centers[:, 0] - radii
    c3 = centers[:, 1] - radii
    c4 = 1.0 - centers[:, 1] - radii
    
    # Pairwise non-overlap constraints (vectorized)
    ci = centers[pair_i]
    cj = centers[pair_j]
    ri = radii[pair_i]
    rj = radii[pair_j]
    
    dist_sq = np.sum((ci - cj)**2, axis=1)
    r_sum_sq = (ri + rj)**2
    
    return np.concatenate([c1, c2, c3, c4, dist_sq - r_sum_sq])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    pair_i, pair_j = np.triu_indices(n, k=1)
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    init_configs = []
    
    # 1. Hexagonal lattice (densest theoretical packing)
    r_h = 0.09
    y = r_h
    row = 0
    pts = []
    while len(pts) < n + 10:
        x = r_h + (row % 2) * r_h
        while x <= 1.0 - r_h and len(pts) < n + 10:
            pts.append([x, y])
            x += 2.0 * r_h
        y += r_h * np.sqrt(3)
        row += 1
    init_configs.append(np.array(pts[:n]))
    
    # 2. Uniform grid with one point shifted to fill gap
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    grid_pts = []
    for y in ys:
        for x in xs:
            grid_pts.append([x, y])
    grid_pts.append([0.5, 0.05])
    init_configs.append(np.array(grid_pts[:n]))
    
    # 3. Controlled random
    np.random.seed(42)
    init_configs.append(np.random.uniform(0.15, 0.85, (n, 2)))
    
    # Optimization loop with multiple restarts
    num_restarts = 12
    for cfg_idx, cfg_centers in enumerate(init_configs):
        for trial in range(num_restarts):
            seed = cfg_idx * num_restarts + trial + 1000
            np.random.seed(seed)
            
            # Perturb initial positions to escape symmetric local minima
            c_pert = cfg_centers.copy()
            c_pert += np.random.uniform(-0.02, 0.02, c_pert.shape)
            c_pert = np.clip(c_pert, 0.05, 0.95)
            
            # Start with a feasible small radius
            r_init = np.full(n, 0.05)
            x0 = np.concatenate([c_pert.flatten(), r_init])
            
            try:
                res = minimize(objective_function, x0, args=(n,), method='SLSQP',
                               bounds=bounds,
                               constraints={'type': 'ineq', 'fun': constraint_function, 'args': (n, pair_i, pair_j)},
                               options={'maxiter': 2500, 'ftol': 1e-12, 'disp': False})
                
                if res.success:
                    c_opt = res.x[:2*n].reshape(n, 2)
                    r_opt = res.x[2*n:]
                    
                    # Quick feasibility check before expensive validation
                    if np.all(r_opt >= -1e-7):
                        # Boundary check
                        if (np.all(c_opt[:, 0] - r_opt >= -1e-9) and 
                            np.all(c_opt[:, 0] + r_opt <= 1.0 + 1e-9) and 
                            np.all(c_opt[:, 1] - r_opt >= -1e-9) and 
                            np.all(c_opt[:, 1] + r_opt <= 1.0 + 1e-9)):
                            
                            # Overlap check
                            dists = np.sqrt(np.sum((c_opt[pair_i] - c_opt[pair_j])**2, axis=1))
                            r_sums = r_opt[pair_i] + r_opt[pair_j]
                            if np.all(dists >= r_sums - 1e-9):
                                s = np.sum(r_opt)
                                if s > best_sum:
                                    best_sum = s
                                    best_centers = c_opt.copy()
                                    best_radii = r_opt.copy()
            except Exception:
                continue
                
    # Fallback if optimization completely fails (highly unlikely)
    if best_centers is None:
        coords = np.linspace(0.1, 0.9, 5)
        centers = []
        count = 0
        for y in coords:
            for x in coords:
                if count < n:
                    centers.append([x, y])
                    count += 1
        while count < n:
            centers.append([0.5, 0.05])
            count += 1
        best_centers = np.array(centers)
        best_radii = np.full(n, 0.08)
        best_sum = np.sum(best_radii)
        
    # Post-processing: Strictly enforce constraints to satisfy validator tolerance
    for i in range(n):
        x, y = best_centers[i]
        best_radii[i] = min(best_radii[i], x, 1-x, y, 1-y)
        
    # Iterative shrink to resolve any remaining pairwise overlaps
    for _ in range(5):
        for i in range(n):
            for j in range(i + 1, n):
                dx = best_centers[i, 0] - best_centers[j, 0]
                dy = best_centers[i, 1] - best_centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = best_radii[i] + best_radii[j]
                if dist < sum_r:
                    shrink = (sum_r - dist) / 2.0 + 1e-8
                    best_radii[i] = max(0.0, best_radii[i] - shrink)
                    best_radii[j] = max(0.0, best_radii[j] - shrink)
                    
    return best_centers, best_radii, float(best_sum)
