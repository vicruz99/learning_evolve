# sol_000040 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 5778b268) state=dc9cf4da sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_smooth_min_clearance(centers, k):
    """
    Computes a smooth approximation of the minimum clearance (to boundaries and other circles).
    Uses log-sum-exp trick for numerical stability.
    """
    n = centers.shape[0]
    
    # Distances to each boundary
    d_bound = np.column_stack([
        centers[:, 0],
        1.0 - centers[:, 0],
        centers[:, 1],
        1.0 - centers[:, 1]
    ])
    
    # Pairwise distances / 2
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    d_pair = dists / 2.0
    
    # Flatten all constraints
    all_vals = np.concatenate([d_bound.ravel(), d_pair.ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    
    # Smooth minimum: -1/k * log(sum(exp(-k * x)))
    # Equivalent to: max(x) - 1/k * log(sum(exp(x - max(x))))
    max_val = np.max(all_vals)
    return max_val - np.log(np.sum(np.exp(all_vals - max_val))) / k

def packing_objective(centers_flat, n, k):
    """Objective function to maximize (negative of smooth min clearance)"""
    centers = centers_flat.reshape(n, 2)
    return -compute_smooth_min_clearance(centers, k)

def run_packing():
    np.random.seed(42)
    n = 26
    bounds = [(0.0, 1.0) for _ in range(2 * n)]
    
    best_val = -np.inf
    best_centers = None
    
    # Generate initial configurations
    configs = []
    
    # 1. Hexagonal lattice initialization
    r_est = 0.1
    pts = []
    for i in range(8):
        for j in range(6):
            x = r_est + j * 2 * r_est + (0.5 if i % 2 == 1 else 0.0) * 2 * r_est
            y = r_est + i * np.sqrt(3) * r_est
            if x <= 1.0 and y <= 1.0:
                pts.append([x, y])
    pts = np.array(pts[:n])
    if len(pts) < n:
        while len(pts) < n:
            pts = np.vstack([pts, np.array([np.random.uniform(0.1, 0.9, 2)])])
    configs.append(pts[:n])
    
    # 2. Perturbed hex lattice
    configs.append(np.clip(configs[0] + np.random.normal(0, 0.03, size=(n, 2)), 0.05, 0.95))
    
    # 3. Random dense initialization
    configs.append(np.random.uniform(0.1, 0.9, size=(n, 2)))
    
    # 4. Regular grid initialization
    cx = np.linspace(0.1, 0.9, 6)
    cy = np.linspace(0.1, 0.9, 5)
    grid_pts = np.array([[x, y] for y in cy for x in cx])[:n]
    configs.append(grid_pts)

    # Optimize each configuration with increasing sharpness k
    for cfg in configs:
        current_centers = cfg.copy()
        for k in [20.0, 50.0, 100.0, 200.0]:
            res = minimize(
                packing_objective,
                current_centers.flatten(),
                args=(n, k),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 3000, 'ftol': 1e-13}
            )
            current_centers = res.x.reshape(n, 2)
            val = compute_smooth_min_clearance(current_centers, k)
            if val > best_val:
                best_val = val
                best_centers = current_centers.copy()
                
    # Compute exact maximal equal radius from optimized centers
    centers = best_centers
    min_r = 1.0
    for i in range(n):
        x, y = centers[i]
        min_r = min(min_r, x, 1.0 - x, y, 1.0 - y)
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            min_r = min(min_r, d / 2.0)
            
    # Apply microscopic safety margin to guarantee strict validity
    min_r *= 0.99999999
    radii = np.full(n, min_r)
    sum_radii = float(np.sum(radii))
    
    return centers, radii, sum_radii
