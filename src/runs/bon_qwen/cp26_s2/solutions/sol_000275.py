# sol_000275 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9068c8d6) state=f6ea10f2 sum of radii=1.291784 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_r = 0.0
    best_centers = np.zeros((n, 2))
    
    # Run multiple trials with different random seeds to find global optimum
    for trial in range(15):
        np.random.seed(trial * 137 + 7)
        
        # Initialize centers in a hexagonal-like grid perturbed by noise
        centers = np.zeros((n, 2))
        idx = 0
        row = 0
        while idx < n:
            x_start = 0.5 * (row % 2) * 0.16 + 0.12
            y_pos = 0.12 + row * 0.16
            for col in range(6):
                if idx >= n: break
                centers[idx, 0] = x_start + col * 0.16
                centers[idx, 1] = y_pos
                idx += 1
            row += 1
            
        centers += np.random.rand(n, 2) * 0.04
        centers = np.clip(centers, 0.05, 0.95)
        
        r = 0.04
        step = 0.004
        
        for it in range(5000):
            # Compute pairwise differences and distances (vectorized)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist_sq = np.maximum(dist_sq, 1e-12)  # Prevent division by zero
            dist = np.sqrt(dist_sq)
            
            # Repulsive force for overlapping circles
            rep = np.where(dist < 2*r, (2*r - dist)/dist, 0.0)
            force = rep[:, :, np.newaxis] * diff
            F = np.sum(force, axis=1)
            
            # Boundary repulsion forces
            F[centers[:, 0] < r, 0] += (r - centers[centers[:, 0] < r, 0])
            F[centers[:, 0] > 1-r, 0] -= (centers[centers[:, 0] > 1-r, 0] - (1-r))
            F[centers[:, 1] < r, 1] += (r - centers[centers[:, 1] < r, 1])
            F[centers[:, 1] > 1-r, 1] -= (centers[centers[:, 1] > 1-r, 1] - (1-r))
            
            # Update positions
            centers += step * F
            centers = np.clip(centers, 1e-6, 1 - 1e-6)
            
            # Check overlap to potentially increase radius
            max_ovl = np.max(np.where(dist < 2*r, 2*r - dist, 0.0))
            if max_ovl < 1e-5:
                r += 0.0002
                
            # Decay step size for convergence
            step *= 0.999
            
            # Occasional random perturbation to escape local minima
            if it % 600 == 0:
                centers += np.random.randn(n, 2) * step * 3.0
                centers = np.clip(centers, 1e-6, 1 - 1e-6)
                
        # Calculate the actual valid radius for this configuration
        min_d = 2.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((centers[i]-centers[j])**2))
                if d < min_d: 
                    min_d = d
                    
        min_bnd = min(np.min(centers[:,0]), np.min(centers[:,1]), 
                      np.min(1-centers[:,0]), np.min(1-centers[:,1]))
        
        curr_r = min(min_d/2, min_bnd)
        
        if curr_r > best_r:
            best_r = curr_r
            best_centers = centers.copy()
            
    radii = np.full(n, best_r)
    return best_centers, radii, float(np.sum(radii))
