# sol_000019 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 38145db4) state=9e0b2144 sum of radii=2.193267 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_sum_radii(centers):
    """Compute the maximum sum of radii for a given set of centers."""
    n = centers.shape[0]
    # Boundary constraints: distance to each side of the unit square
    r = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
    r = np.minimum(r, np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Pairwise constraints: half the distance to the nearest other center
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    r = np.minimum(r, min_dists * 0.5)
    
    return np.sum(r)

def objective(x):
    """Objective function to minimize (negative sum of radii)."""
    centers = x.reshape(-1, 2)
    return -compute_sum_radii(centers)

def run_packing():
    np.random.seed(42)
    n = 26
    best_val = -np.inf
    best_centers = None
    
    # Generate multiple starting configurations to avoid local optima
    starts = []
    
    # 1. Hexagonal lattice arrangement (promotes dense packing)
    pts = []
    for i in range(6):
        for j in range(6):
            x = j * np.sqrt(3)/2 + (np.sqrt(3)/4 if i % 2 else 0)
            y = i * 0.75
            pts.append([x, y])
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = np.array(pts[:n])
    # Normalize to [0, 1] and apply a margin to keep circles inside initially
    pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0) + 1e-9)
    pts = pts * 0.85 + 0.075
    starts.append(pts.ravel())
    
    # 2. Random initialization
    starts.append(np.random.rand(n * 2))
    
    # 3. Perturbed grid initialization
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 6), np.linspace(0.1, 0.9, 6))
    grid = np.vstack((gx.ravel(), gy.ravel())).T
    grid = grid[:n]
    grid += np.random.randn(n, 2) * 0.05
    starts.append(grid.ravel())
    
    # Run local optimization from each start
    for x0 in starts:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 15000, 'xatol': 1e-7, 'fatol': 1e-7})
        current_val = compute_sum_radii(res.x.reshape(-1, 2))
        if current_val > best_val:
            best_val = current_val
            best_centers = res.x.reshape(-1, 2)
            
    # Compute final radii for the best configuration
    r = np.minimum(best_centers[:, 0], 1.0 - best_centers[:, 0])
    r = np.minimum(r, np.minimum(best_centers[:, 1], 1.0 - best_centers[:, 1]))
    diffs = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    r = np.minimum(r, min_dists * 0.5)
    
    return best_centers, r, np.sum(r)
