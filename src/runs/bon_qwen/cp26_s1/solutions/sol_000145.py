# sol_000145 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a66096c7) state=462affa8 sum of radii=2.008680 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def spread_penalty(vars, n):
    """
    Computes the objective value for packing optimization.
    Minimizes -d + sum of penalties for boundary and overlap violations.
    """
    centers = vars[:n*2].reshape(n, 2)
    d = vars[n*2]
    obj = -d  # We want to maximize d
    half_d = d * 0.5
    
    # Boundary penalties: circles must stay at least half_d away from walls
    obj += np.sum(np.maximum(0.0, half_d - centers[:, 0])**2)
    obj += np.sum(np.maximum(0.0, centers[:, 0] + half_d - 1.0)**2)
    obj += np.sum(np.maximum(0.0, half_d - centers[:, 1])**2)
    obj += np.sum(np.maximum(0.0, centers[:, 1] + half_d - 1.0)**2)
    
    # Pairwise distance penalties: circles must be at least d apart
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1.0)  # Ignore self-distance
    obj += np.sum(np.maximum(0.0, d - dists)**2)
    
    return obj

def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Initialize centers with a perturbed hexagonal lattice pattern
    # Row distribution [5, 6, 5, 6, 4] approximates optimal hex packing density
    centers = []
    counts = [5, 6, 5, 6, 4]
    y = 0.12
    dy = 0.18
    for i, c in enumerate(counts):
        dx = 0.16
        x_start = (1.0 - (c - 1) * dx) / 2.0
        if i % 2 == 1:
            x_start += dx / 2.0  # Shift odd rows for hexagonal packing
        for j in range(c):
            x = x_start + j * dx
            # Add small noise to break symmetry and help optimization
            centers.append([x + np.random.normal(0, 0.005), 
                            y + np.random.normal(0, 0.005)])
        y += dy
        
    centers = np.array(centers)
    centers = np.clip(centers, 0.02, 0.98)
    
    # 2. Estimate initial separation d based on current layout
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_boundary_dist = min(np.min(centers[:, 0]), np.min(1.0 - centers[:, 0]),
                            np.min(centers[:, 1]), np.min(1.0 - centers[:, 1]))
    d_init = min(min_pair_dist, 2.0 * min_boundary_dist) * 0.95
    
    x0 = np.concatenate([centers.ravel(), [d_init]])
    
    # 3. Optimize to maximize minimum separation distance
    bounds = [(0.0, 1.0)] * (n * 2) + [(0.01, 0.5)]
    res = minimize(spread_penalty, x0, args=(n,), method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
                   
    best_centers = res.x[:n * 2].reshape(n, 2)
    
    # 4. Compute strictly valid radius from optimized positions
    diffs = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    min_bdry = min(np.min(best_centers[:, 0]), np.min(1.0 - best_centers[:, 0]),
                   np.min(best_centers[:, 1]), np.min(1.0 - best_centers[:, 1]))
                   
    # Radius is half the limiting distance
    r = min(min_pair, 2.0 * min_bdry) / 2.0
    radii = np.full(n, r)
    
    return best_centers, radii, float(np.sum(radii))
