# sol_000311 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1cbfbe8a) state=f3df539d sum of radii=2.622469 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(centers, radii):
    """Compute boundary and overlap penalty for the current configuration."""
    n = len(radii)
    pen = 0.0
    
    # Boundary penalties
    d_left = radii - centers[:, 0]
    pen += np.sum(np.maximum(d_left, 0.0)**2)
    d_right = radii - (1.0 - centers[:, 0])
    pen += np.sum(np.maximum(d_right, 0.0)**2)
    d_bot = radii - centers[:, 1]
    pen += np.sum(np.maximum(d_bot, 0.0)**2)
    d_top = radii - (1.0 - centers[:, 1])
    pen += np.sum(np.maximum(d_top, 0.0)**2)
    
    # Overlap penalties
    dx = centers[:, None, 0] - centers[None, :, 0]
    dy = centers[:, None, 1] - centers[None, :, 1]
    dist = np.sqrt(dx**2 + dy**2) + 1e-12
    min_dist = radii[:, None] + radii[None, :]
    
    # Only consider lower triangle to avoid double counting
    mask = np.tril(np.ones((n, n), dtype=bool), -1)
    diff = min_dist[mask] - dist[mask]
    pen += np.sum(np.maximum(diff, 0.0)**2)
    
    return pen

def objective(params, n, lam):
    """Objective function: minimize negative sum of radii plus weighted penalty."""
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    return -np.sum(radii) + lam * compute_penalty(centers, radii)

def run_packing():
    """Optimize circle packing to maximize sum of radii."""
    np.random.seed(42)
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Different initialization strategies
    restart_configs = [
        {"type": "hex", "rows": [5, 5, 5, 5, 6]},
        {"type": "grid", "rows": None},
        {"type": "rand", "rows": None}
    ]
    
    for cfg in restart_configs:
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.08)
        
        # Generate initial layout
        if cfg["type"] == "hex":
            idx = 0
            y = 0.0
            s = 0.15
            for r_i, cnt in enumerate(cfg["rows"]):
                x_off = 0.0 if r_i % 2 == 0 else s / 2.0
                for c_i in range(cnt):
                    centers[idx, 0] = x_off + c_i * s
                    centers[idx, 1] = y
                    idx += 1
                y += s * np.sqrt(3.0) / 2.0
            # Normalize to fit comfortably inside [0,1]
            centers = (centers - centers.min(0)) / (centers.max(0) - centers.min(0)) * 0.9 + 0.05
        elif cfg["type"] == "grid":
            idx = 0
            for i in range(5):
                for j in range(5):
                    centers[idx, 0] = 0.1 + j * 0.2
                    centers[idx, 1] = 0.1 + i * 0.2
                    idx += 1
            centers[25, :] = [0.5, 0.5]
        else:
            centers = np.random.rand(n, 2) * 0.8 + 0.1
            
        params = np.concatenate([centers.ravel(), radii])
        bounds = [(0.0, 1.0)] * (2 * n) + [(1e-5, 0.5)] * n
        
        # Annealing schedule: gradually enforce constraints
        lambdas = [10.0, 50.0, 200.0, 1000.0, 5000.0]
        curr_params = params.copy()
        
        for lam in lambdas:
            res = minimize(objective, curr_params, args=(n, lam), method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 250, 'ftol': 1e-10})
            curr_params = res.x
            
        centers_opt = curr_params[:2*n].reshape(n, 2)
        radii_opt = curr_params[2*n:]
        
        # Strict repair step to guarantee validation passes
        min_gap = 1.0
        for i in range(n):
            r = radii_opt[i]
            min_gap = min(min_gap, centers_opt[i, 0] - r, 1.0 - centers_opt[i, 0] - r,
                          centers_opt[i, 1] - r, 1.0 - centers_opt[i, 1] - r)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
                min_gap = min(min_gap, d - (radii_opt[i] + radii_opt[j]))
                
        if min_gap < 0:
            radii_opt += min_gap
            radii_opt = np.maximum(radii_opt, 1e-9)
            
        curr_sum = np.sum(radii_opt)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = centers_opt.copy()
            best_radii = radii_opt.copy()
            
    return best_centers, best_radii, best_sum
