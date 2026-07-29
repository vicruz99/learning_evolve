# sol_000287 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a4dfceb8) state=bbb90b46 sum of radii=1.972251 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def optimize_packing(centers, n, radii):
    """
    Iteratively expands radii and resolves overlaps using repulsion forces.
    """
    i_idx, j_idx = np.triu_indices(n, k=1)
    lr = 0.008
    momentum = 0.85
    vel = np.zeros_like(centers)
    
    for step in range(400):
        # Compute pairwise distances efficiently
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Maximum feasible radius based on boundaries and neighbors
        bnd = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                         np.minimum(centers[:, 1], 1 - centers[:, 1]))
        nbr = 0.5 * np.min(dists, axis=1)
        
        # Update radii with smoothing to prevent oscillations
        new_radii = np.minimum(bnd, nbr)
        radii = np.maximum(new_radii, radii * 0.95)
        
        # Gradient steps to resolve overlaps and optimize positions
        for _ in range(30):
            force = np.zeros_like(centers)
            
            di = centers[i_idx]
            dj = centers[j_idx]
            diff = di - dj
            d = np.linalg.norm(diff, axis=1)
            d_safe = np.maximum(d, 1e-6)
            dir_vec = diff / d_safe[:, np.newaxis]
            
            overlaps = radii[i_idx] + radii[j_idx] - d
            mask = overlaps > 1e-7
            if np.any(mask):
                f_mag = overlaps[mask]
                f_dir = dir_vec[mask]
                np.add.at(force, i_idx[mask], -f_dir * f_mag[:, np.newaxis])
                np.add.at(force, j_idx[mask], f_dir * f_mag[:, np.newaxis])
                
            # Boundary repulsion forces
            for dim in range(2):
                left_ovl = np.maximum(0, radii - centers[:, dim])
                mask_l = left_ovl > 1e-7
                if np.any(mask_l):
                    force[mask_l, dim] -= left_ovl[mask_l]
                right_ovl = np.maximum(0, centers[:, dim] + radii - 1)
                mask_r = right_ovl > 1e-7
                if np.any(mask_r):
                    force[mask_r, dim] -= right_ovl[mask_r]
                    
            # Velocity update with momentum
            vel = momentum * vel + lr * force
            centers += vel
            centers = np.clip(centers, 1e-5, 1 - 1e-5)
            
        # Decay learning rate for fine-tuning
        if step % 30 == 0 and step > 0:
            lr *= 0.95
            
        # Occasional jitter to escape local minima
        if step % 50 == 0:
            centers += np.random.uniform(-0.002, 0.002, centers.shape)
            centers = np.clip(centers, 0.01, 0.99)
            
    return centers, radii, np.sum(radii)

def run_packing():
    np.random.seed(42)
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Generate diverse initial configurations
    configs = []
    
    # 1. Square grid
    c1 = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < n:
                c1[idx, 0] = 0.15 + j * 0.17
                c1[idx, 1] = 0.15 + i * 0.17
                idx += 1
    if n > 25:
        c1[25, 0] = 0.5
        c1[25, 1] = 0.5
    configs.append(c1)
    
    # 2. Hexagonal lattice
    c2 = np.zeros((n, 2))
    idx = 0
    y = 0.12
    row = 0
    while idx < n:
        x = 0.15 if row % 2 == 0 else 0.25
        while x <= 0.85 and idx < n:
            c2[idx, 0] = x
            c2[idx, 1] = y
            idx += 1
            x += 0.2
        y += 0.173
        row += 1
    configs.append(c2)
    
    # 3. Random perturbation
    configs.append(np.random.uniform(0.1, 0.9, (n, 2)))
    
    # Evaluate each configuration
    for init_c in configs:
        radii = np.full(n, 0.04)
        init_c += np.random.uniform(-0.005, 0.005, init_c.shape)
        init_c = np.clip(init_c, 0.05, 0.95)
        
        cent, rad, s = optimize_packing(init_c.copy(), n, radii)
        if s > best_sum:
            best_sum = s
            best_centers = cent
            best_radii = rad
            
    # Final strict projection to guarantee validity
    diffs = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    bnd = np.minimum(np.minimum(best_centers[:, 0], 1 - best_centers[:, 0]),
                     np.minimum(best_centers[:, 1], 1 - best_centers[:, 1]))
    nbr = 0.5 * np.min(dists, axis=1)
    best_radii = np.minimum(bnd, nbr)
    
    return best_centers, best_radii, np.sum(best_radii)
