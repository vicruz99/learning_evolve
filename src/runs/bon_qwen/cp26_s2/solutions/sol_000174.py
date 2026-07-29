# sol_000174 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ae65bcc8) state=86e45c49 sum of radii=2.429097 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_radii(centers):
    """Compute maximum feasible radii for given centers."""
    n = centers.shape[0]
    radii = np.full(n, np.inf)
    
    # Distance to boundaries
    radii = np.minimum(radii, centers[:, 0])
    radii = np.minimum(radii, 1.0 - centers[:, 0])
    radii = np.minimum(radii, centers[:, 1])
    radii = np.minimum(radii, 1.0 - centers[:, 1])
    
    # Distance to other circles (pairwise)
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    radii = np.minimum(radii, np.min(dists, axis=1) / 2.0)
    
    return radii

def objective(centers_flat):
    """Objective function to minimize (negative sum of radii)."""
    centers = centers_flat.reshape(-1, 2)
    radii = get_radii(centers)
    return -np.sum(radii)

def run_packing():
    n = 26
    best_centers = None
    best_sum = 0.0
    
    np.random.seed(42)
    
    # Prepare diverse initial configurations to avoid poor local optima
    inits = []
    
    # 1. Hexagonal lattice arrangement (dense packing structure)
    hex_pts = []
    for i in range(8):
        for j in range(6):
            x = j * 1.0 + (0.5 if i % 2 else 0)
            y = i * np.sqrt(3)/2 * 1.0
            hex_pts.append([x, y])
        if len(hex_pts) >= n:
            break
    hex_arr = np.array(hex_pts[:n])
    mx, my = hex_arr.max(axis=0) - hex_arr.min(axis=0)
    s = 0.8 / max(mx, my)
    hex_arr = (hex_arr - hex_arr.min(axis=0)) * s + 0.1
    inits.append(hex_arr)
    
    # 2. 5x5 grid + center point (uniform structure)
    grid = np.linspace(0.15, 0.85, 5)
    pts = np.array(np.meshgrid(grid, grid)).T.reshape(-1, 2)
    extra = np.array([[0.5, 0.5]])
    inits.append(np.vstack([pts, extra]))
    
    # 3. Random uniform distribution
    inits.append(np.random.uniform(0.1, 0.9, (n, 2)))
    
    # 4. Clustered around quadrant centers
    clust = []
    for cx, cy in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]:
        clust.append(np.random.normal([cx, cy], 0.08, (7, 2)))
    clust_arr = np.vstack(clust)
    clust_arr = np.clip(clust_arr, 0.05, 0.95)
    inits.append(clust_arr)

    # Local optimization from multiple starts
    for init in inits:
        for _ in range(2):
            # Perturb initialization to escape symmetric traps
            x0 = init.copy() + np.random.normal(0, 0.015, init.shape)
            x0 = np.clip(x0, 0.02, 0.98)
            
            # Nelder-Mead is robust for non-smooth objectives
            res = minimize(objective, x0.flatten(), method='Nelder-Mead',
                           options={'maxiter': 30000, 'xatol': 1e-7, 'fatol': 1e-9})
            
            cur_sum = -res.fun
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_centers = res.x.reshape(-1, 2)
                
    # Final high-precision refinement
    if best_centers is not None:
        res = minimize(objective, best_centers.flatten(), method='Powell',
                       options={'maxiter': 40000, 'xtol': 1e-8, 'ftol': 1e-10})
        if -res.fun > best_sum:
            best_sum = -res.fun
            best_centers = res.x.reshape(-1, 2)
            
    # Enforce strict boundaries and recompute exact feasible radii
    best_centers = np.clip(best_centers, 0.0, 1.0)
    best_radii = get_radii(best_centers)
    
    return best_centers, best_radii, float(best_sum)
