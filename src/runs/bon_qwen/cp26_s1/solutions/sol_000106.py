# sol_000106 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a3c1a30f) state=a2937308 sum of radii=2.204487 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_function(centers_flat):
    """
    Computes the negative sum of maximal feasible radii for given centers.
    Used as the objective for minimization.
    """
    centers = centers_flat.reshape(26, 2)
    centers = np.clip(centers, 0.0, 1.0)
    
    # Distance to boundaries: min(x, 1-x, y, 1-y)
    dists_bnd = np.array([centers[:, 0], 1.0 - centers[:, 0],
                          centers[:, 1], 1.0 - centers[:, 1]])
    min_bnd = np.min(dists_bnd, axis=0)
    
    # Pairwise distances between all circles
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists_mat, np.inf)  # Ignore self-distance
    min_pair = np.min(dists_mat, axis=1)
    
    # Maximal radius is half the minimum distance to any obstacle
    radii = np.minimum(min_bnd, 0.5 * min_pair)
    return -np.sum(radii)

def compute_final_radii(centers):
    """
    Computes the exact maximal radii for a given set of centers.
    """
    centers = np.clip(centers, 0.0, 1.0)
    dists_bnd = np.array([centers[:, 0], 1.0 - centers[:, 0],
                          centers[:, 1], 1.0 - centers[:, 1]])
    min_bnd = np.min(dists_bnd, axis=0)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_mat = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists_mat, np.inf)
    min_pair = np.min(dists_mat, axis=1)
    
    return np.minimum(min_bnd, 0.5 * min_pair)

def run_packing():
    np.random.seed(42)
    best_score = -np.inf
    best_centers = None
    
    # Generate diverse initial configurations
    init_configs = []
    
    # 1. Pure random starts
    for _ in range(4):
        init_configs.append(np.random.rand(26, 2))
        
    # 2. Random clustered start (encourages exploration of denser regions)
    init_configs.append(np.random.rand(26, 2) * 0.6 + 0.2)
    
    # 3. Structured grid start (good baseline for hexagonal-like packing)
    grid = np.array(np.meshgrid(np.linspace(0.15, 0.85, 6), np.linspace(0.15, 0.85, 5))).T.reshape(-1, 2)
    init_configs.append(grid[:26])
    
    # 4. Quasi-random (low-discrepancy) start
    idx = np.arange(1, 27)
    x_qr = (np.mod(idx * 0.618033988749895, 1.0))
    y_qr = (np.mod(idx * 0.381966011250105, 1.0))
    init_configs.append(np.column_stack([x_qr, y_qr]) * 0.8 + 0.1)
    
    # Run optimization from each start
    for start in init_configs:
        x0 = start.flatten()
        res = minimize(objective_function, x0, method='Nelder-Mead',
                       options={'maxiter': 8000, 'xatol': 1e-7, 'fatol': 1e-9})
        score = -res.fun
        if score > best_score:
            best_score = score
            best_centers = res.x.reshape(26, 2)
            
    # Final precise radius computation and validation consistency
    final_centers = best_centers
    final_radii = compute_final_radii(final_centers)
    final_sum = float(np.sum(final_radii))
    
    return final_centers, final_radii, final_sum
