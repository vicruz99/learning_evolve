# sol_000023 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1c033854) state=c7ddf2c9 sum of radii=1.880073 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    N = 26
    
    # Initialize centers in a hexagonal lattice pattern
    centers = np.zeros((N, 2))
    idx = 0
    rows = [6, 5, 6, 5, 4] # 26 circles total
    R_init = 0.08
    s = (1.0 - 2 * R_init) / 5.0
    y_step = s * np.sqrt(3) / 2.0
    
    for r_idx, count in enumerate(rows):
        y = R_init + r_idx * y_step
        x_start = R_init + (6 - count) * s / 2.0
        for c_idx in range(count):
            x = x_start + c_idx * s
            centers[idx] = [x, y]
            idx += 1
            
    R = R_init
    lr = 0.003
    steps = 12000
    
    for step in range(steps):
        # Slowly expand the target radius to push the packing limit
        R += 0.000005
        
        # Vectorized distance computation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        
        # Compute repulsive forces for overlapping circles
        overlap = np.maximum(0.0, 2 * R - dist)
        force_mag = overlap
        
        safe_dist = np.where(dist > 1e-8, dist, 1.0)
        dir_vec = diff / safe_dist[:, :, np.newaxis]
        
        forces_pair = force_mag[:, :, np.newaxis] * dir_vec
        forces = np.sum(forces_pair, axis=1)
        
        # Boundary repulsion forces
        dx_b = np.where(centers[:, 0] < R, R - centers[:, 0], 
                        np.where(centers[:, 0] > 1 - R, (1 - R) - centers[:, 0], 0.0))
        dy_b = np.where(centers[:, 1] < R, R - centers[:, 1], 
                        np.where(centers[:, 1] > 1 - R, (1 - R) - centers[:, 1], 0.0))
        
        forces[:, 0] += dx_b
        forces[:, 1] += dy_b
        
        # Update positions
        centers += lr * forces
        
        # Enforce boundary constraints strictly
        centers[:, 0] = np.clip(centers[:, 0], R, 1 - R)
        centers[:, 1] = np.clip(centers[:, 1], R, 1 - R)
        
        # Decay learning rate for finer adjustments near convergence
        lr *= 0.9995
        
    # Compute exact maximum feasible radii for each circle given the final centers
    # This guarantees a valid packing and maximizes sum of radii for these positions
    radii = np.full(N, 2.0)
    for i in range(N):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        for j in range(N):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                max_r = min(max_r, d / 2)
        radii[i] = max_r
        
    sum_r = np.sum(radii)
    return centers, radii, sum_r
