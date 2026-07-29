# sol_000356 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b8d6b6a1) state=8ea1dcb5 sum of radii=1.134097 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Initialize centers on a perturbed grid covering the square
    x_vals = np.linspace(0.15, 0.85, 6)
    y_vals = np.linspace(0.15, 0.85, 5)
    centers = np.array([[x, y] for x in x_vals for y in y_vals][:n])
    centers += np.random.uniform(-0.03, 0.03, centers.shape)
    centers = np.clip(centers, 0.1, 0.9)
    
    r = 0.10
    lr = 0.008
    
    # Iterative relaxation to pack circles and grow radius
    for step in range(5000):
        # Compute pairwise distances efficiently
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        forces = np.zeros_like(centers)
        
        # 1. Pairwise overlap repulsion
        overlaps = 2.0 * r - dists
        overlaps = np.where(overlaps > 0, overlaps, 0.0)
        
        inv_dists = np.where(dists > 1e-8, 1.0 / dists, 0.0)
        directions = diff * inv_dists[:, :, None]
        
        pair_forces = overlaps[:, :, None] * directions
        forces += np.sum(pair_forces, axis=1)
        
        # 2. Boundary repulsion (soft walls)
        wall_k = 30.0
        forces[:, 0] += wall_k * np.maximum(0, r - centers[:, 0])
        forces[:, 0] -= wall_k * np.maximum(0, centers[:, 0] - (1 - r))
        forces[:, 1] += wall_k * np.maximum(0, r - centers[:, 1])
        forces[:, 1] -= wall_k * np.maximum(0, centers[:, 1] - (1 - r))
        
        # Update positions
        centers += lr * forces
        
        # Hard clip to prevent numerical drift outside [0,1]
        centers[:, 0] = np.clip(centers[:, 0], 1e-7, 1 - 1e-7)
        centers[:, 1] = np.clip(centers[:, 1], 1e-7, 1 - 1e-7)
        
        # Evaluate slack to decide radius growth
        min_pair_dist = np.min(dists)
        margins = np.concatenate([
            centers[:, 0] - r, 1 - centers[:, 0] - r,
            centers[:, 1] - r, 1 - centers[:, 1] - r
        ])
        min_margin = np.min(margins)
        slack = min(min_pair_dist - 2.0 * r, min_margin)
        
        if slack > 1e-4:
            r += 0.0004
            lr = 0.008  # Maintain step size when growing
        elif slack < -1e-4:
            lr *= 0.95  # Slow down to resolve overlaps
        else:
            lr *= 0.99
            
        # Occasional noise to escape local minima
        if step % 800 == 0 and step > 0:
            centers += np.random.normal(0, 0.008, centers.shape)
            centers = np.clip(centers, 0.05, 0.95)
            
    # Final strict validation and radius adjustment
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    margins = np.concatenate([
        centers[:, 0], 1 - centers[:, 0], centers[:, 1], 1 - centers[:, 1]
    ])
    min_margin = np.min(margins)
    
    # Ensure strict validity by taking the feasible maximum radius
    feasible_r = min(min_pair_dist / 2.0, min_margin)
    feasible_r = max(0.0, feasible_r)
    
    radii = np.full(n, feasible_r)
    sum_radii = float(np.sum(radii))
    
    return centers, radii, sum_radii
