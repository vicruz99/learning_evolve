# sol_000087 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000046 (state 0aa7241c) state=a6ae05e3 sum of radii=2.268595 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_smooth_min(centers, k=60.0):
    """
    Computes a smooth approximation of the minimum clearance to boundaries 
    and half-pairwise distances using log-sum-exp.
    """
    n = centers.shape[0]
    # Distance to boundaries
    b = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    b = np.minimum(b, centers[:, 1])
    b = np.minimum(b, 1.0 - centers[:, 1])
    
    # Half pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    p = dists[np.triu_indices(n, k=1)] / 2.0
    
    vals = np.concatenate([b, p])
    v_max = np.max(vals)
    # Stable computation of smooth minimum
    exp_vals = np.exp(-k * (vals - v_max))
    return v_max - np.log(np.sum(exp_vals)) / k

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2 * n)
    
    def objective(vars_flat):
        c = vars_flat.reshape(n, 2)
        # Maximize minimum clearance => minimize negative smooth min
        return -compute_smooth_min(c, k=60.0)
        
    best_val = np.inf
    best_centers = None
    
    configs = []
    # Diverse hexagonal lattice initializations known to pack densely
    for rows in [[6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 5, 5, 6], [4, 6, 6, 6, 4], [6, 6, 5, 5, 4]]:
        r0 = 0.095
        y = r0
        row_idx = 0
        pts = []
        for cnt in rows:
            shift = r0 if row_idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) < n: 
                    pts.append([x, y])
                x += 2 * r0
            y += np.sqrt(3) * r0
            row_idx += 1
        configs.append(np.array(pts[:n]))
        
    # Generate perturbed versions to escape local minima and symmetry traps
    np.random.seed(42)
    base_configs = configs[:4]
    for cfg in base_configs:
        for _ in range(4):
            p = cfg + np.random.uniform(-0.025, 0.025, cfg.shape)
            p = np.clip(p, 0.05, 0.95)
            configs.append(p)
            
    # Optimize from each configuration using L-BFGS-B
    for cfg in configs:
        try:
            res = minimize(objective, cfg.flatten(), method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 20000, 'ftol': 1e-13, 'gtol': 1e-11})
            if res.fun < best_val:
                best_val = res.fun
                best_centers = res.x.reshape(n, 2)
        except Exception:
            continue
            
    # Fallback safety
    if best_centers is None:
        best_centers = configs[0]
        
    # Compute exact maximum feasible equal radius for the optimized layout
    c = best_centers
    min_r = np.minimum(np.minimum(c[:,0], 1.0 - c[:,0]), np.minimum(c[:,1], 1.0 - c[:,1]))
    
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_r = np.minimum(min_r, np.min(dists, axis=1) / 2.0)
    
    # Take the global minimum clearance and apply a tiny safety buffer for numerical validation
    R = np.min(min_r) - 1e-9
    radii = np.full(n, max(R, 1e-9))
    sum_r = float(np.sum(radii))
    
    return best_centers, radii, sum_r
