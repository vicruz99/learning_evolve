# sol_000024 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 764eb384) state=c504814b sum of radii=1.978600 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    n = 26
    # 1. Hexagonal lattice initialization
    centers = np.zeros((n, 2))
    idx = 0
    spacing = 0.18
    for row in range(6):
        for col in range(6):
            if idx >= n:
                break
            x = col * spacing + (row % 2) * (spacing / 2) + 0.05
            y = row * (spacing * np.sqrt(3) / 2) + 0.05
            centers[idx] = [x, y]
            idx += 1
            
    r = 0.06
    lr = 0.02
    max_ov_threshold = 1e-5
    r_step = 0.00005
    stagnation = 0
    boundary_k = 30.0
    
    # 2. Main optimization loop
    for it in range(200000):
        # Vectorized pairwise differences and distances
        diffs = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Overlap calculation (positive means overlapping)
        overlap = 2.0 * r - dists
        overlap = np.clip(overlap, 0.0, None)
        
        # Direction vectors (avoid division by zero on diagonal)
        dirs = diffs / dists[:, :, None]
        
        # Repulsive forces from overlaps
        forces = np.sum(overlap[:, :, None] * dirs, axis=1)
        
        # Boundary repulsion
        forces[:, 0] += np.maximum(0.0, r - centers[:, 0]) * boundary_k
        forces[:, 0] -= np.maximum(0.0, centers[:, 0] - (1.0 - r)) * boundary_k
        forces[:, 1] += np.maximum(0.0, r - centers[:, 1]) * boundary_k
        forces[:, 1] -= np.maximum(0.0, centers[:, 1] - (1.0 - r)) * boundary_k
        
        # Update positions
        centers += lr * forces
        centers = np.clip(centers, 0.0, 1.0)
        
        # Learning rate decay
        lr *= 0.9995
        
        # Schedule: increase radius if configuration is stable
        if it % 300 == 0:
            max_ov = np.max(overlap)
            if max_ov < max_ov_threshold:
                r += r_step
                lr = 0.02  # Reset learning rate for new constraint scale
                stagnation = 0
                # Small random perturbation to escape local minima
                centers += np.random.randn(n, 2) * 0.0005
            else:
                stagnation += 300
                
        if stagnation > 15000:
            break
            
    # 3. Final polishing phase with higher boundary stiffness
    for _ in range(30000):
        diffs = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        overlap = np.clip(2.0 * r - dists, 0.0, None)
        dirs = diffs / dists[:, :, None]
        forces = np.sum(overlap[:, :, None] * dirs, axis=1)
        
        forces[:, 0] += np.maximum(0.0, r - centers[:, 0]) * 100.0
        forces[:, 0] -= np.maximum(0.0, centers[:, 0] - (1.0 - r)) * 100.0
        forces[:, 1] += np.maximum(0.0, r - centers[:, 1]) * 100.0
        forces[:, 1] -= np.maximum(0.0, centers[:, 1] - (1.0 - r)) * 100.0
        
        centers += 0.008 * forces
        centers = np.clip(centers, 0.0, 1.0)
        
    radii = np.full(n, r)
    return centers, radii, np.sum(radii)
