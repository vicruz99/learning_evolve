# sol_000230 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2fe8b400) state=c4b37d8d sum of radii=2.609736 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_function(vars, n, mu):
    """
    Computes the penalized objective for circle packing.
    vars: array of shape (3*n,) -> [x1, y1, r1, x2, y2, r2, ...]
    n: number of circles
    mu: penalty weight
    """
    X = vars[:2*n].reshape(n, 2)
    R = vars[2*n:]

    # Boundary penalties: circles must stay inside [0,1]x[0,1]
    p_left   = np.maximum(0, R - X[:, 0])**2
    p_right  = np.maximum(0, R - (1.0 - X[:, 0]))**2
    p_bottom = np.maximum(0, R - X[:, 1])**2
    p_top    = np.maximum(0, R - (1.0 - X[:, 1]))**2
    bound_pen = np.sum(p_left + p_right + p_bottom + p_top)

    # Pairwise overlap penalties
    # Compute distance matrix efficiently
    dx = X[:, 0, None] - X[:, 0]
    dy = X[:, 1, None] - X[:, 1]
    dist = np.sqrt(dx**2 + dy**2 + 1e-8)
    
    # Sum of radii matrix
    r_sum = R[:, None] + R[None, :]
    
    # Overlap is positive when circles intersect
    overlap = np.maximum(0, r_sum - dist)
    
    # Sum squared overlaps for upper triangle only (i < j)
    triu_idx = np.triu_indices(n, k=1)
    pair_pen = np.sum(overlap[triu_idx]**2)

    # Objective: maximize sum of radii <=> minimize -sum(r)
    return -np.sum(R) + mu * (bound_pen + pair_pen)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_vars = None
    best_sum = -np.inf

    # Optimization settings
    n_restarts = 3
    n_epochs = 12
    mu_start = 5.0
    mu_growth = 1.6

    for seed in range(n_restarts):
        np.random.seed(seed)
        
        # Initialize with grid + perturbation
        v = np.zeros(3 * n)
        for i in range(n):
            col = i % 5
            row = i // 5
            # Base grid positions
            v[3*i]     = 0.05 + 0.2 * col
            v[3*i+1]   = 0.05 + 0.2 * row
            v[3*i+2]   = 0.05
            # Add noise to break symmetry
            v[3*i]     += np.random.randn() * 0.02
            v[3*i+1]   += np.random.randn() * 0.02
            
        # Ensure initial guess respects bounds
        v[:2*n] = np.clip(v[:2*n], 0.0, 1.0)
        v[2*n:] = np.clip(v[2*n:], 0.0, 0.5)

        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        mu = mu_start

        for _ in range(n_epochs):
            res = minimize(
                objective_function, 
                v, 
                args=(n, mu), 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            v = res.x
            mu *= mu_growth

        # Evaluate candidate
        R = v[2*n:]
        X = v[:2*n].reshape(n, 2)
        
        if np.all(R >= 0):
            current_sum = np.sum(R)
            if current_sum > best_sum:
                best_sum = current_sum
                best_vars = v.copy()

    # Extract best result
    centers = best_vars[:2*n].reshape(n, 2)
    radii = best_vars[2*n:]

    # Strict post-processing to guarantee validation passes
    # 1. Enforce boundary constraints
    radii = np.minimum(radii, np.minimum(np.minimum(centers[:,0], 1-centers[:,0]), 
                                         np.minimum(centers[:,1], 1-centers[:,1])))
    centers[:, 0] = np.clip(centers[:, 0], radii, 1 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1 - radii)

    # 2. Resolve pairwise overlaps by symmetric shrinking
    for _ in range(30):
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j]:
                    overlap = (radii[i] + radii[j] - dist) / 2.0
                    radii[i] = max(0.0, radii[i] - overlap)
                    radii[j] = max(0.0, radii[j] - overlap)

    return centers, radii, float(np.sum(radii))
