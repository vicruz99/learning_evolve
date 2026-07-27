# sol_000027 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 27de0ea1) state=1a18139f sum of radii=1.153614 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n = 26
    centers = np.zeros((n, 2))
    
    # 1. Initial placement: 5x5 grid + 1 extra circle, slightly perturbed
    grid = np.linspace(0.1, 0.9, 5)
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [grid[j], grid[i]]
            idx += 1
    centers[25] = [0.5, 0.5]
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    r = 0.02
    dt = 0.02
    force_scale = 30.0
    boundary_scale = 80.0
    max_iter = 12000
    
    for step in range(max_iter):
        # 2. Vectorized pairwise repulsive forces
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Identify overlapping pairs
        overlap_mask = dists < 2*r + 1e-8
        safe_dists = np.where(dists < 1e-8, 1e-8, dists)
        
        # Compute repulsion magnitude and vector
        f_mag = np.where(overlap_mask, (2*r - dists) * force_scale, 0.0)
        f_vecs = diffs / safe_dists[:, :, np.newaxis] * f_mag[:, :, np.newaxis]
        
        # Sum forces for each particle (antisymmetry ensures correct net force)
        forces = np.sum(f_vecs, axis=1)
        
        # 3. Boundary repulsive forces
        # Left wall
        mask_left = centers[:, 0] < r
        forces[mask_left, 0] += (r - centers[mask_left, 0]) * boundary_scale
        # Right wall
        mask_right = centers[:, 0] > 1 - r
        forces[mask_right, 0] -= (centers[mask_right, 0] - (1 - r)) * boundary_scale
        # Bottom wall
        mask_bottom = centers[:, 1] < r
        forces[mask_bottom, 1] += (r - centers[mask_bottom, 1]) * boundary_scale
        # Top wall
        mask_top = centers[:, 1] > 1 - r
        forces[mask_top, 1] -= (centers[mask_top, 1] - (1 - r)) * boundary_scale
        
        # Update positions
        centers += dt * forces
        centers = np.clip(centers, 0, 1)
        
        # 4. Adaptive radius growth with validation
        if step % 10 == 0:
            valid = True
            # Check boundaries
            if np.any(centers[:, 0] < r - 1e-5) or np.any(centers[:, 0] > 1 - r + 1e-5) or \
               np.any(centers[:, 1] < r - 1e-5) or np.any(centers[:, 1] > 1 - r + 1e-5):
                valid = False
            # Check overlaps
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                        if d < 2*r - 1e-5:
                            valid = False
                            break
                    if not valid: break
                    
            if valid:
                r += 0.0003
            else:
                r += 0.00001 # Slow creep to explore boundaries
                
        # Decay learning rate for stability
        dt *= 0.9997
        
    # 5. Final extraction of feasible radius
    min_d = 2.0
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i]-centers[j])**2))
            if d < min_d: min_d = d
        d_b = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        if d_b < min_d: min_d = d_b
        
    # Subtract small epsilon to satisfy strict validation constraints
    r_final = min_d / 2.0 - 1e-6
    radii = np.full(n, r_final)
    
    return centers, radii, float(np.sum(radii))
