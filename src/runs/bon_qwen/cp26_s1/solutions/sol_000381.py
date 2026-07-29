# sol_000381 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7a0a6c4a) state=e8c9d31b sum of radii=2.463519 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_energy(centers, r):
    """Compute overlap and boundary violation energy for fixed radius r."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    mask = np.triu(np.ones((26, 26), dtype=bool), k=1)
    overlaps = np.maximum(0.0, 2.0 * r - dists)
    energy = np.sum(overlaps[mask] ** 2)
    
    for i in range(26):
        for c in range(2):
            if centers[i, c] < r:
                energy += (r - centers[i, c]) ** 2
            if centers[i, c] > 1.0 - r:
                energy += (centers[i, c] - (1.0 - r)) ** 2
    return energy

def objective_func(x, r_val):
    """Wrapper for scipy optimizer."""
    return compute_energy(x.reshape(26, 2), r_val)

def run_packing():
    np.random.seed(42)
    
    # 1. Initialize on a hexagonal lattice (6-5-6-5-4 pattern)
    points = []
    for row in range(5):
        if row == 4:
            n_cols = 4
        elif row % 2 == 0:
            n_cols = 6
        else:
            n_cols = 5
        for col in range(n_cols):
            x = col * 2.0 + (row % 2) * 1.0
            y = row * np.sqrt(3)
            points.append([x, y])
    points = np.array(points)
    
    # Normalize and center in [0, 1] with initial margin
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    bbox_size = max_pt - min_pt
    scale = 0.85 / np.max(bbox_size)
    centers = (points - min_pt) * scale + (1.0 - scale * bbox_size) / 2.0
    
    # Add slight random perturbation to break symmetry
    centers += np.random.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    r = 0.08
    # 2. Iteratively optimize positions and increase radius
    for _ in range(100):
        res = minimize(
            objective_func, 
            centers.flatten(), 
            args=(r,), 
            method='BFGS', 
            options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
        )
        centers = res.x.reshape(26, 2)
        
        if compute_energy(centers, r) < 1e-9:
            r += 0.0006
        else:
            if r > 0.135:
                break
            continue
            
    # 3. Final validity check and safety buffer
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    min_d = np.min(dists[np.triu_indices(26, k=1)])
    
    boundary_min = np.min(centers)
    boundary_max = np.max(centers)
    max_r_center = min(boundary_min, 1.0 - boundary_max)
    
    # Determine strictly feasible radius
    r_final = min(r, min_d / 2.0, max_r_center)
    r_final = max(r_final - 1e-7, 0.0)
    
    # Clamp centers to stay inside [r, 1-r]
    centers = np.clip(centers, r_final, 1.0 - r_final)
    radii = np.full(26, r_final)
    
    return centers, radii, np.sum(radii)
