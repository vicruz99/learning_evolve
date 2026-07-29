# sol_000151 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 46a34d55) state=5c9a7c59 sum of radii=1.011139 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_clearance(centers):
    """Compute the maximum feasible equal radius for a given set of centers."""
    n = centers.shape[0]
    # Boundary clearance: distance to the closest wall
    b = np.min(np.minimum(centers, 1 - centers))
    
    # Pairwise clearance: half the minimum distance between centers
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = np.tril(np.ones((n, n), dtype=bool), -1)
    p = np.min(dists[mask]) / 2.0 if np.any(mask) else float('inf')
    
    return min(b, p)

def compute_forces(centers, r):
    """Compute repulsive forces for overlapping circles and boundary violations."""
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    mask = np.tril(np.ones((n, n), dtype=bool), -1)
    
    # Overlap amount: positive when dist < 2r
    overlaps = np.maximum(0.0, 2*r - dists)
    overlaps = np.where(mask, overlaps, 0.0)
    
    # Unit vectors pointing from j to i
    safe_dists = np.where(dists > 1e-8, dists, 1e-8)
    unit_vec = diff / safe_dists[:, :, None]
    
    # Force vectors: magnitude = overlap, direction = apart
    force_vec = unit_vec * overlaps[:, :, None]
    
    # Aggregate forces: i gets +F, j gets -F
    forces = np.zeros_like(centers)
    forces += np.sum(force_vec, axis=1)
    forces -= np.sum(force_vec, axis=0)
    
    # Boundary forces: push circles inside if they cross r or 1-r
    for d in range(2):
        mask_l = centers[:, d] < r
        forces[mask_l, d] += r - centers[mask_l, d]
        mask_r = centers[:, d] > 1 - r
        forces[mask_r, d] -= centers[mask_r, d] - (1 - r)
        
    return forces

def run_packing():
    n = 26
    best_r = 0
    best_centers = None
    
    # Multiple restarts to ensure finding the global optimum
    for seed in range(3):
        np.random.seed(seed + 42)
        
        # Initialize in a hexagonal pattern
        centers = np.zeros((n, 2))
        idx = 0
        r_init = 0.08
        for row in range(6):
            y = r_init + row * r_init * 1.732
            if y > 1 - r_init: break
            shift = (row % 2) * r_init
            for col in range(7):
                x = r_init + shift + col * 2 * r_init
                if x > 1 - r_init: break
                if idx < n:
                    centers[idx] = [x, y]
                    idx += 1
        # Fill remaining slots randomly if needed
        while idx < n:
            centers[idx] = [np.random.rand(), np.random.rand()]
            idx += 1
            
        # Force-directed relaxation with growing radius
        r = 0.05
        lr = 0.005
        for step in range(3000):
            forces = compute_forces(centers, r)
            centers += forces * lr
            centers = np.clip(centers, 1e-6, 1 - 1e-6)
            
            r += 0.00005
            lr *= 0.998  # Adaptive decay for stability
            
        # Evaluate final clearance
        cur_r = compute_clearance(centers)
        if cur_r > best_r:
            best_r = cur_r
            best_centers = centers.copy()
            
    # Slight safety margin to satisfy strict numerical tolerances
    final_r = best_r - 1e-10
    radii = np.full(n, final_r)
    return best_centers, radii, np.sum(radii)
