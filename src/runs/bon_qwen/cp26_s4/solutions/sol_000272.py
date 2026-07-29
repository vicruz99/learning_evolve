# sol_000272 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c64acbd5) state=f0368f51 sum of radii=0.651259 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    n = 26
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    best_sum = 0.0
    
    # Try multiple random initializations to find the global optimum
    for seed in range(10):
        np.random.seed(seed)
        centers = np.random.rand(n, 2)
        r = 0.02
        radii = np.full(n, r)
        
        steps = 10000
        k = 150.0 # Repulsion strength
        
        for step in range(steps):
            # Adaptive step size that decays over time
            dt = 0.006 / (1.0 + step / 4000.0)
            
            # Compute pairwise differences and distances
            diff = centers[:, None, :] - centers[None, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            
            # Safe distance for normalization
            dist_safe = np.maximum(dist, 1e-9)
            dir_vec = diff / dist_safe[:, :, None]
            
            # Mask for upper triangle (unique pairs)
            mask = np.triu(np.ones((n, n)), k=1) > 0
            
            # Repulsive forces between overlapping circles
            min_d = 2.0 * r
            overlap = np.where(mask, np.maximum(0, min_d - dist), 0.0)
            forces = np.sum(overlap[:, :, None] * dir_vec * k, axis=1)
            
            # Repulsive forces from walls
            wf = np.zeros_like(centers)
            wf[:, 0] += np.maximum(0, r - centers[:, 0]) * k
            wf[:, 0] -= np.maximum(0, centers[:, 0] - (1 - r)) * k
            wf[:, 1] += np.maximum(0, r - centers[:, 1]) * k
            wf[:, 1] -= np.maximum(0, centers[:, 1] - (1 - r)) * k
            
            forces += wf
            
            # Update centers
            centers += forces * dt
            
            # Strictly enforce boundary constraints
            centers[:, 0] = np.clip(centers[:, 0], r, 1 - r)
            centers[:, 1] = np.clip(centers[:, 1], r, 1 - r)
            
            # Gradually increase radius to densify packing
            if step % 5 == 0:
                # Calculate maximum possible radius based on current positions
                # Distance to walls
                w_dist = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                                    np.minimum(centers[:, 1], 1 - centers[:, 1]))
                min_w = np.min(w_dist)
                
                # Distance between circles
                min_c = np.min(dist[mask])
                
                max_r = min(min_w, min_c / 2.0)
                
                # Smoothly grow radius
                if max_r > r:
                    r = min(max_r, r * 1.0015)
                radii[:] = r
                
            # Add small jitter occasionally to escape local minima
            if step % 1000 == 0 and step > 0:
                centers += np.random.randn(n, 2) * 0.002
                centers[:, 0] = np.clip(centers[:, 0], r, 1 - r)
                centers[:, 1] = np.clip(centers[:, 1], r, 1 - r)
                
        current_sum = n * r
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
    return best_centers, best_radii, best_sum
