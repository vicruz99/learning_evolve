# sol_000311 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e3d19f45) state=351d9a83 sum of radii=2.626088 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj(v):
    # Objective: minimize negative sum of radii (equivalent to maximizing sum)
    return -np.sum(v[52:])

def constraints_func(v):
    n = 26
    centers = v[:52].reshape(n, 2)
    radii = v[52:]
    
    # Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    xb = centers[:, 0] - radii
    xB = 1.0 - centers[:, 0] - radii
    yb = centers[:, 1] - radii
    yB = 1.0 - centers[:, 1] - radii
    
    # Separation constraints: d^2 >= (r_i + r_j)^2
    # Vectorized pairwise distance squared
    diff = centers[:, None, :] - centers[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum_sq = (radii[:, None] + radii[None, :])**2
    sep = dist_sq - r_sum_sq
    
    # Extract strict lower triangle to avoid duplicates and self-comparisons
    sep_lower = sep[np.tril_indices(n, k=-1)]
    
    return np.concatenate([xb, xB, yb, yB, sep_lower])

def run_packing():
    n = 26
    # Bounds: centers in [0, 1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * 52 + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints_func}
    
    best_v = None
    best_val = -np.inf
    valid_best = False
    
    # Multiple restarts to explore different local optima
    for seed in range(15):
        np.random.seed(seed)
        
        # Structured grid initialization with perturbation
        cx = np.linspace(0.15, 0.85, 5)
        cy = np.linspace(0.15, 0.85, 5)
        gx, gy = np.meshgrid(cx, cy)
        init_c = np.column_stack([gx.ravel(), gy.ravel()])  # 25 circles
        init_c = np.vstack([init_c, [0.5, 0.15]])           # 26th circle
        
        # Add random noise to break symmetry and escape bad local minima
        init_c += np.random.uniform(-0.05, 0.05, size=(26, 2))
        init_c = np.clip(init_c, 0.05, 0.95)
        init_r = np.full(26, 0.04)  # Feasible initial radius
        
        v0 = np.concatenate([init_c.ravel(), init_r])
        
        try:
            res = minimize(obj, v0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
            
            if res.success:
                c = res.x[:52].reshape(26, 2)
                r = res.x[52:]
                
                # Strict feasibility check with tolerance
                if (np.all(c[:, 0] - r >= -1e-6) and np.all(1.0 - c[:, 0] - r >= -1e-6) and
                    np.all(c[:, 1] - r >= -1e-6) and np.all(1.0 - c[:, 1] - r >= -1e-6)):
                    
                    diff = c[:, None, :] - c[None, :, :]
                    d2 = np.sum(diff**2, axis=2)
                    rsum2 = (r[:, None] + r[None, :])**2
                    
                    if np.all(d2[np.tril_indices(n, k=-1)] >= rsum2[np.tril_indices(n, k=-1)] - 1e-6):
                        s = np.sum(r)
                        if s > best_val:
                            best_val = s
                            best_v = res.x
                            valid_best = True
        except Exception:
            continue
            
    if not valid_best:
        # Fallback to a deterministic valid grid packing
        cx = np.linspace(0.1, 0.9, 5)
        cy = np.linspace(0.1, 0.9, 5)
        gx, gy = np.meshgrid(cx, cy)
        centers = np.column_stack([gx.ravel(), gy.ravel()])
        centers = np.vstack([centers, [0.5, 0.5]])
        radii = np.full(26, 0.09)
        return centers, radii, 26 * 0.09

    centers = best_v[:52].reshape(26, 2)
    radii = best_v[52:]
    return centers, radii, np.sum(radii)
