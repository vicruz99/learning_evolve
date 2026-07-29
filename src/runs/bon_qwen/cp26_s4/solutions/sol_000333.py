# sol_000333 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 488bfafc) state=16f6d1b7 sum of radii=2.512292 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n, penalty_weight):
    """
    Computes the objective: maximize sum of radii subject to packing constraints.
    Uses a penalty method for non-overlap and boundary constraints.
    """
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    # Negative sum of radii (we minimize this to maximize sum)
    obj = -np.sum(radii)
    
    # Vectorized pairwise distance and overlap calculation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    req = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Only consider lower triangle to avoid double counting and self-distance
    mask = np.tril(np.ones((n, n), dtype=bool), -1)
    overlaps = np.maximum(0.0, req[mask] - dists[mask])
    obj += penalty_weight * np.sum(overlaps**2)
    
    # Vectorized boundary penalties
    x, y = centers[:, 0], centers[:, 1]
    r = radii
    obj += penalty_weight * np.sum(np.maximum(0.0, r - x)**2)
    obj += penalty_weight * np.sum(np.maximum(0.0, x + r - 1.0)**2)
    obj += penalty_weight * np.sum(np.maximum(0.0, r - y)**2)
    obj += penalty_weight * np.sum(np.maximum(0.0, y + r - 1.0)**2)
    
    return obj

def run_packing():
    n = 26
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # Define multiple initial configurations to escape local minima
    initial_configs = []
    
    # 1. Random initialization
    np.random.seed(123)
    initial_configs.append(np.random.uniform(0.2, 0.8, (n, 2)))
    
    # 2. Grid initialization (5x5 + 1)
    grid_pts = np.array(np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))).T.reshape(-1, 2)
    initial_configs.append(np.vstack([grid_pts, [0.5, 0.5]]))
    
    # 3. Hexagonal-inspired initialization
    hex_pts = []
    for row in range(5):
        y = 0.1 + row * 0.2
        num_cols = 6 if row % 2 == 1 else 5
        offset = 0.05 if row % 2 == 1 else 0.0
        for col in range(num_cols):
            x = 0.1 + col * 0.18 + offset
            hex_pts.append([x, y])
    initial_configs.append(np.array(hex_pts)[:n])

    # Optimization bounds
    bounds = [(0.0, 1.0)] * (2*n) + [(1e-6, 0.5)] * n
    
    for centers_init in initial_configs:
        radii_init = np.full(n, 0.06)
        vars_init = np.concatenate([centers_init.flatten(), radii_init])
        
        # Stage 1: Moderate penalty to find a good region
        res1 = minimize(compute_objective, vars_init, args=(n, 1000.0),
                        method='L-BFGS-B', bounds=bounds, options={'maxiter': 1500, 'ftol': 1e-9})
        
        # Stage 2: High penalty to strictly enforce constraints
        res2 = minimize(compute_objective, res1.x, args=(n, 10000.0),
                        method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
                        
        centers_cand = res2.x[:2*n].reshape(n, 2)
        radii_cand = res2.x[2*n:]
        
        # Validation check
        valid = True
        for i in range(n):
            x, y = centers_cand[i]
            r = radii_cand[i]
            if x - r < -1e-9 or x + r > 1.0 + 1e-9 or y - r < -1e-9 or y + r > 1.0 + 1e-9:
                valid = False
                break
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers_cand[i] - centers_cand[j])**2))
                if d < radii_cand[i] + radii_cand[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            s = np.sum(radii_cand)
            if s > best_sum_radii:
                best_sum_radii = s
                best_centers = centers_cand.copy()
                best_radii = radii_cand.copy()
        else:
            # If invalid due to numerical tolerance, scale down radii slightly to guarantee validity
            scale = 1.0
            while not valid and scale > 0.1:
                scale *= 0.99
                radii_temp = radii_cand * scale
                valid = True
                for i in range(n):
                    x, y = centers_cand[i]
                    r = radii_temp[i]
                    if x < r or x > 1.0 - r or y < r or y > 1.0 - r:
                        valid = False; break
                    for j in range(i + 1, n):
                        if np.linalg.norm(centers_cand[i] - centers_cand[j]) < radii_temp[i] + radii_temp[j]:
                            valid = False; break
                    if not valid: break
            if valid:
                s = np.sum(radii_temp)
                if s > best_sum_radii:
                    best_sum_radii = s
                    best_centers = centers_cand.copy()
                    best_radii = radii_temp
                    
    return best_centers, best_radii, best_sum_radii
