# sol_000082 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000032 (state ac51bd1a) state=57907d1c sum of radii=1.403813 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_penalty(vars_arr, n, mu):
    """
    Computes the objective: negative target radius + penalty for constraint violations.
    Optimizes equal-radius packing by maximizing a common radius t.
    """
    cx = vars_arr[:n]
    cy = vars_arr[n:2*n]
    t = vars_arr[2*n]
    
    # Objective: maximize t => minimize -t
    obj = -t
    
    penalty = 0.0
    
    # Boundary penalties: t <= x <= 1-t, t <= y <= 1-t
    penalty += np.sum(np.maximum(0, t - cx)**2)
    penalty += np.sum(np.maximum(0, t - (1.0 - cx))**2)
    penalty += np.sum(np.maximum(0, t - cy)**2)
    penalty += np.sum(np.maximum(0, t - (1.0 - cy))**2)
    
    # Overlap penalties: dist(i,j) >= 2*t
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    
    triu_idx = np.triu_indices(n, k=1)
    dist_upper = dist[triu_idx]
    penalty += np.sum(np.maximum(0, 2.0 * t - dist_upper)**2)
    
    return obj + mu * penalty

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)
    
    best_t = 0.0
    best_centers = None
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal pattern with row counts [5, 6, 5, 5, 5] (total 26)
    pts = []
    r_est = 0.1
    dy = np.sqrt(3) * r_est
    for row in range(5):
        y = row * dy
        cnt = 6 if row == 1 else 5
        shift = r_est if row % 2 == 1 else 0.0
        x_start = (6 - cnt) * r_est  # Center the row within width
        for i in range(cnt):
            pts.append([x_start + i * 2 * r_est + shift, y])
    pts = np.array(pts)
    # Scale and shift to fit comfortably in [0,1]
    pts -= pts.min(axis=0)
    pts /= pts.max(axis=0)
    pts = pts * 0.85 + 0.075
    inits.append(pts)
    
    # 2. Perturbed hexagonal to break symmetry
    inits.append(pts + np.random.uniform(-0.04, 0.04, pts.shape))
    
    # 3. Multiple random valid starts to escape local minima
    for i in range(15):
        rng = np.random.default_rng(i * 7 + 1)
        c = rng.uniform(0.15, 0.85, (n, 2))
        inits.append(c)
        
    # Variable bounds: centers in [0.05, 0.95], radius t in [0.08, 0.12]
    bounds = [(0.05, 0.95)] * (2 * n) + [(0.08, 0.12)]
    
    # Optimization loop over all initial configurations
    for c_init in inits:
        c_init = np.clip(c_init, 0.05, 0.95)
        x0 = np.concatenate([c_init.flatten(), [0.098]])
        
        try:
            res = minimize(
                objective_penalty,
                x0,
                args=(n, 5000.0),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-14}
            )
            
            if np.isfinite(res.fun):
                t_opt = res.x[2*n]
                centers_cand = res.x[:2*n].reshape(n, 2)
                
                # Quick validity check to filter out failed runs
                min_wall = min(np.min(centers_cand), np.min(1.0 - centers_cand))
                dists = np.linalg.norm(centers_cand[:, None] - centers_cand[None, :], axis=2)
                np.fill_diagonal(dists, np.inf)
                min_pair = np.min(dists) / 2.0
                feasible_t = min(min_wall, min_pair)
                
                if feasible_t > best_t:
                    best_t = feasible_t
                    best_centers = centers_cand.copy()
        except Exception:
            continue
            
    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = inits[0]
        best_t = 0.09
        
    # Post-processing: Compute exact maximum radius for each circle individually.
    # This extracts the maximum possible sum of radii for the optimized centers,
    # potentially allowing some circles to be larger than others if space permits.
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundaries
        d_wall = min(best_centers[i, 0], 1.0 - best_centers[i, 0],
                     best_centers[i, 1], 1.0 - best_centers[i, 1])
        
        # Distance to other centers
        diffs = best_centers - best_centers[i]
        d_pairs = np.sqrt(np.sum(diffs**2, axis=1))
        d_pairs[i] = np.inf
        d_pair_min = np.min(d_pairs)
        
        # Maximum valid radius for this circle
        radii[i] = min(d_wall, d_pair_min / 2.0)
        
    # Apply a tiny safety margin for numerical stability in the validator
    radii *= 0.999995
    sum_radii = np.sum(radii)
    
    return best_centers, radii, sum_radii
