# sol_000165 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 724447fa) state=831c60c8 sum of radii=2.334554 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_loss(vars):
    """
    Computes the negative sum of radii plus penalty terms for boundary and overlap violations.
    Vectorized for performance.
    """
    n = N_CIRCLES
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Boundary penalties: circles must stay within [0,1]^2
    # Penalty is squared violation of r <= coord and r <= 1 - coord
    bx = np.maximum(0.0, r - c[:, 0])**2
    bx1 = np.maximum(0.0, r - (1.0 - c[:, 0]))**2
    by = np.maximum(0.0, r - c[:, 1])**2
    by1 = np.maximum(0.0, r - (1.0 - c[:, 1]))**2
    bound_pen = np.sum(bx + bx1 + by + by1)
    
    # Overlap penalties: distance between centers must be >= sum of radii
    # Compute pairwise distances using broadcasting
    dx = c[:, 0, np.newaxis] - c[np.newaxis, :, 0]
    dy = c[:, 1, np.newaxis] - c[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Extract upper triangle indices to avoid double counting and self-comparison
    idx = np.triu_indices(n, k=1)
    overlaps = r_sum[idx] - dists[idx]
    overlap_pen = np.sum(np.maximum(0.0, overlaps)**2)
    
    # Objective: maximize sum of radii -> minimize negative sum
    # Penalty weight is high to enforce constraints strictly
    return -np.sum(r) + 10000.0 * (bound_pen + overlap_pen)

def run_packing():
    n = N_CIRCLES
    
    # Initialize with a hexagonal grid pattern for better convergence
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.06)
    
    idx = 0
    for i in range(6):
        ncols = 5 if i % 2 == 0 else 4
        for j in range(ncols):
            if idx >= n: break
            # Hexagonal spacing
            x = 0.15 + j * 0.17 + (i % 2) * 0.085
            y = 0.10 + i * 0.15
            centers[idx] = [x, y]
            idx += 1
        if idx >= n: break
            
    # Add random perturbation to break symmetry and aid optimization
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Flatten for optimizer: [x1, y1, ..., x26, y26, r1, ..., r26]
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: coordinates in [0, 1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    best_res = None
    best_val = np.inf
    
    # Multi-start optimization to find the best local minimum
    for _ in range(4):
        x0_pert = x0 + np.random.normal(0, 0.01, x0.shape)
        x0_pert = np.clip(x0_pert, [b[0] for b in bounds], [b[1] for b in bounds])
        
        res = minimize(compute_loss, x0_pert, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9})
        
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
            
    c_opt = best_res.x[:2*n].reshape(n, 2)
    r_opt = best_res.x[2*n:]
    
    # Post-processing to guarantee strict feasibility within validator tolerance
    # Iteratively shrink overlapping circles
    for _ in range(30):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(c_opt[i,0]-c_opt[j,0], c_opt[i,1]-c_opt[j,1])
                if d < r_opt[i] + r_opt[j]:
                    shrink = (r_opt[i] + r_opt[j] - d) / 2.0 + 1e-9
                    r_opt[i] = max(0.0, r_opt[i] - shrink)
                    r_opt[j] = max(0.0, r_opt[j] - shrink)
                    changed = True
        if not changed:
            break
            
    # Enforce boundary constraints strictly
    for i in range(n):
        x, y = c_opt[i]
        r = r_opt[i]
        r_opt[i] = min(r, x, 1.0-x, y, 1.0-y)
        r_opt[i] = max(0.0, r_opt[i] - 1e-9)

    return c_opt, r_opt, float(np.sum(r_opt))
