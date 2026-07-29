# sol_000174 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 397104b2) state=58dca5e7 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Initialization: Perturbed grid layout
    centers = []
    for i in range(5):
        for j in range(5):
            centers.append([0.15 + j * 0.14, 0.15 + i * 0.14])
    centers.append([0.5, 0.5])  # 26th circle
    centers = np.array(centers)
    centers += np.random.randn(*centers.shape) * 0.015
    centers = np.clip(centers, 0.1, 0.9)
    
    radii = np.full(n, 0.02)
    
    # 2. Simulation parameters
    dt = 0.005
    k_rep = 6.0
    k_exp = 0.00025
    max_iter = 6000
    
    for step in range(max_iter):
        # Expand radii
        radii += k_exp
        
        # Vectorized pairwise repulsion
        diffs = centers[:, None] - centers[None, :]          # (N, N, 2)
        dists = np.linalg.norm(diffs, axis=2)                # (N, N)
        np.fill_diagonal(dists, np.inf)                      # Ignore self-interaction
        
        min_dists = radii[:, None] + radii[None, :]
        overlaps = np.maximum(0.0, min_dists - dists)
        
        # Safe division to avoid NaNs when dists are near zero
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        dirs = diffs / safe_dists[:, :, None]
        
        # Accumulate forces from all pairs
        forces = (overlaps[:, :, None] * k_rep * dirs).sum(axis=1)
        
        # Boundary repulsion forces
        forces[:, 0] += np.where(centers[:, 0] < radii, k_rep * (radii - centers[:, 0]), 0.0)
        forces[:, 0] -= np.where(centers[:, 0] > 1 - radii, k_rep * (centers[:, 0] - (1 - radii)), 0.0)
        forces[:, 1] += np.where(centers[:, 1] < radii, k_rep * (radii - centers[:, 1]), 0.0)
        forces[:, 1] -= np.where(centers[:, 1] > 1 - radii, k_rep * (centers[:, 1] - (1 - radii)), 0.0)
        
        # Update positions
        centers += dt * forces
        centers = np.clip(centers, 0.0, 1.0)
        
        # Adaptive cooling to settle configuration
        if step % 800 == 0:
            k_exp *= 0.8
            k_rep *= 1.15
            
    # 3. Strict Constraint Projection
    # Enforce boundaries exactly
    radii[:] = np.minimum(
        radii,
        np.minimum(
            np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
            np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        )
    )
    
    # Iteratively resolve pairwise overlaps
    for _ in range(20):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                if radii[i] + radii[j] > d:
                    overlap = radii[i] + radii[j] - d
                    reduction = overlap / 2.0
                    radii[i] -= reduction
                    radii[j] -= reduction
                    
    radii = np.maximum(radii, 0.0)
    
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
