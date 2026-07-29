# sol_000218 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state df9a626f) state=517fa093 sum of radii=2.177690 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    np.random.seed(42)
    
    # 1. Initial hexagonal-ish placement for high density start
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(6):
        for col in range(6):
            if idx >= n: break
            x = 0.12 + col * 0.16 + (0.08 if row % 2 else 0)
            y = 0.12 + row * 0.16
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers[idx] = [x, y]
                idx += 1
    while idx < n:
        centers[idx] = np.random.rand(2) * 0.6 + 0.2
        idx += 1
        
    radii = np.ones(n) * 0.085
    
    # 2. Phase-based expansion and relaxation
    phases = 80
    steps_per_phase = 1000
    base_lr = 0.018
    growth = 0.0025
    
    for phase in range(phases):
        lr = base_lr * (0.97 ** phase)
        radii += growth * (0.92 ** phase)
        
        # Small perturbation to escape local minima
        centers += np.random.randn(n, 2) * lr * 0.1
        
        for step in range(steps_per_phase):
            # Pairwise repulsive forces (vectorized)
            diffs = centers[:, None, :] - centers[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            dists[np.eye(n, dtype=bool)] = np.inf
            
            min_dists = radii[:, None] + radii[None, :]
            overlap = np.maximum(0, min_dists - dists)
            
            safe_dists = np.maximum(dists, 1e-6)
            dirs = diffs / safe_dists[:, :, None]
            force_mags = overlap * 25.0
            f_vec = dirs * force_mags[:, :, None]
            forces = np.sum(f_vec, axis=1)
            
            # Vectorized boundary repulsion forces
            left_ov = np.maximum(0, radii - centers[:, 0])
            right_ov = np.maximum(0, centers[:, 0] + radii - 1)
            bottom_ov = np.maximum(0, radii - centers[:, 1])
            top_ov = np.maximum(0, centers[:, 1] + radii - 1)
            
            forces[:, 0] += (left_ov - right_ov) * 60.0
            forces[:, 1] += (bottom_ov - top_ov) * 60.0
            
            centers += forces * lr
            centers = np.clip(centers, 1e-4, 1-1e-4)
            
    # 3. Final validation and safety adjustment to guarantee strict feasibility
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    dists[np.eye(n, dtype=bool)] = np.inf
    min_dists = radii[:, None] + radii[None, :]
    max_ov = np.max(np.maximum(0, min_dists - dists))
    if max_ov > 1e-6:
        radii -= max_ov / 2.0
        
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1-x, y, 1-y)
        
    return centers, radii, float(np.sum(radii))
