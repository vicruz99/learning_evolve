# sol_000071 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 34f92e2c) state=65832705 sum of radii=2.089360 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed optimization with iterative radius expansion.
    """
    np.random.seed(42)
    n = 26
    
    # 1. Initialization: Hexagonal lattice pattern
    centers = np.zeros((n, 2))
    r_init = 0.08
    idx = 0
    rows_config = [6, 5, 6, 5, 4]
    spacing = 2.0 * r_init
    height = spacing * np.sqrt(3) / 2.0
    
    for row_idx, count in enumerate(rows_config):
        y = (row_idx + 0.5) * height + r_init
        x_start = r_init + (spacing / 2.0) if row_idx % 2 == 1 else r_init
        for col in range(count):
            if idx < n:
                centers[idx, 0] = x_start + col * spacing
                centers[idx, 1] = y
                idx += 1
                
    # Normalize initial positions to fit comfortably inside the square
    min_c = centers.min(axis=0)
    max_c = centers.max(axis=0)
    centers = (centers - min_c) / (max_c - min_c) * 0.8 + 0.1
    
    radii = np.full(n, r_init)
    current_r = r_init
    lr = 0.08
    r_inc = 1.5e-6
    
    # 2. Main Optimization Loop
    for step in range(120000):
        radii[:] = current_r
        
        # Compute Forces
        forces = np.zeros_like(centers)
        
        # Boundary Forces
        # Left
        mask_l = centers[:, 0] < radii
        forces[mask_l, 0] += (radii[mask_l] - centers[mask_l, 0]) * 10.0
        # Right
        mask_r = centers[:, 0] > 1 - radii
        forces[mask_r, 0] -= (centers[mask_r, 0] - (1 - radii[mask_r])) * 10.0
        # Bottom
        mask_b = centers[:, 1] < radii
        forces[mask_b, 1] += (radii[mask_b] - centers[mask_b, 1]) * 10.0
        # Top
        mask_t = centers[:, 1] > 1 - radii
        forces[mask_t, 1] -= (centers[mask_t, 1] - (1 - radii[mask_t])) * 10.0
        
        # Inter-circle Repulsion Forces (Vectorized)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        overlaps = (radii[:, np.newaxis] + radii[np.newaxis, :]) - dists
        overlaps = np.maximum(0, overlaps)
        
        safe_dists = np.maximum(dists, 1e-9)
        # Force direction * magnitude
        forces += np.sum((diff / safe_dists[:, :, np.newaxis]) * overlaps[:, :, np.newaxis], axis=1) * 5.0
        
        # Update Centers
        centers += forces * lr
        np.clip(centers, radii[:, None], 1 - radii[:, None], out=centers)
        
        # Adaptive Control
        if step % 500 == 0:
            current_r += r_inc
            lr *= 0.995  # Cooling schedule
            
        # Random Jitter to escape local minima
        if step % 2000 == 0:
            centers += np.random.normal(0, 0.002, centers.shape)
            np.clip(centers, radii[:, None], 1 - radii[:, None], out=centers)
            
    # 3. Final Strict Resolution
    for _ in range(15000):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        overlaps = np.maximum(0, (radii[:, np.newaxis] + radii[np.newaxis, :]) - dists)
        safe_dists = np.maximum(dists, 1e-9)
        forces = np.sum((diff / safe_dists[:, :, np.newaxis]) * overlaps[:, :, np.newaxis], axis=1) * 3.0
        centers += forces * 0.03
        np.clip(centers, radii[:, None], 1 - radii[:, None], out=centers)
        
    # 4. Validation and Fallback Correction
    def is_valid():
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x < r - 1e-12 or x > 1 - r + 1e-12 or y < r - 1e-12 or y > 1 - r + 1e-12:
                return False
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d < radii[i] + radii[j] - 1e-12:
                    return False
        return True

    if not is_valid():
        # Gradually shrink radii until valid
        for _ in range(2000):
            radii -= 2e-5
            # Quick resolution pass
            for _ in range(200):
                diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                np.fill_diagonal(dists, np.inf)
                ov = np.maximum(0, (radii[:, np.newaxis] + radii[np.newaxis, :]) - dists)
                sd = np.maximum(dists, 1e-9)
                f = np.sum((diff / sd[:, :, np.newaxis]) * ov[:, :, np.newaxis], axis=1) * 5.0
                centers += f * 0.1
                np.clip(centers, radii[:, None], 1 - radii[:, None], out=centers)
            if is_valid():
                break
                
    return centers, radii, float(np.sum(radii))
