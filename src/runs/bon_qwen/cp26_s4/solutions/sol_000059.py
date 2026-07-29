# sol_000059 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 26e3ad40) state=16cdf3a5 sum of radii=1.732040 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(v, n):
    """
    Objective function for circle packing optimization.
    Minimizes negative sum of radii plus quadratic penalty for constraint violations.
    """
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    
    # Boundary penalty: circles must stay inside [0,1]^2
    pen = np.sum(np.maximum(0.0, r - x)**2)
    pen += np.sum(np.maximum(0.0, r - (1.0 - x))**2)
    pen += np.sum(np.maximum(0.0, r - y)**2)
    pen += np.sum(np.maximum(0.0, r - (1.0 - y))**2)
    
    # Pairwise overlap penalty
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d = np.sqrt(dx*dx + dy*dy)
    
    # Set diagonal distance to 1.0 to avoid self-overlap penalties
    np.fill_diagonal(d, 1.0)
    
    r_sum = r[:, None] + r[None, :]
    overlap = np.maximum(0.0, r_sum - d)
    np.fill_diagonal(overlap, 0.0)
    
    pen += np.sum(overlap**2)
    
    # Maximize sum of radii => minimize negative sum + large penalty
    return -np.sum(r) + 20000.0 * pen

def run_packing():
    np.random.seed(42)
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Run multiple trials with different perturbations to escape local minima
    for trial in range(15):
        # Initialize with a structured grid layout
        pos = np.zeros((n, 2))
        idx = 0
        for i in range(6):
            for j in range(5):
                if idx >= n:
                    break
                pos[idx] = [0.05 + j * 0.16, 0.05 + i * 0.14]
                idx += 1
                
        # Add controlled random jitter to break symmetry
        pos += np.random.randn(n, 2) * 0.03
        pos = np.clip(pos, 0.1, 0.9)
        
        rad = np.full(n, 0.05)
        
        # Flatten variables for optimizer: [x1..x26, y1..y26, r1..r26]
        vars0 = np.concatenate([pos[:, 0], pos[:, 1], rad])
        bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
        
        # L-BFGS-B handles bound constraints efficiently
        res = minimize(objective_func, vars0, args=(n,), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 8000, 'ftol': 1e-15, 'gtol': 1e-15})
        
        x_opt = res.x[:n]
        y_opt = res.x[n:2*n]
        r_opt = res.x[2*n:]
        
        # --- Strict Feasibility Projection ---
        # 1. Enforce boundary constraints
        r_opt = np.minimum(r_opt, np.minimum(x_opt, 1.0 - x_opt))
        r_opt = np.minimum(r_opt, np.minimum(y_opt, 1.0 - y_opt))
        
        # 2. Enforce pairwise non-overlap by iteratively shrinking
        for _ in range(50):
            any_overlap = False
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt((x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2)
                    if r_opt[i] + r_opt[j] > dist - 1e-8:
                        shrink = (r_opt[i] + r_opt[j] - dist + 1e-8) / 2.0
                        r_opt[i] -= shrink
                        r_opt[j] -= shrink
                        any_overlap = True
            if not any_overlap:
                break
                
        curr_sum = np.sum(r_opt)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = np.column_stack([x_opt, y_opt])
            best_radii = r_opt.copy()
            
    return best_centers, best_radii, best_sum
