# sol_000158 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 39a6f529) state=eeed6755 sum of radii=1.054637 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    np.random.seed(42)
    
    # Initialize centers and radii
    centers = np.random.uniform(0.15, 0.85, (N, 2))
    radii = np.full(N, 0.02)
    
    dt = 0.025
    steps = 5000
    
    for step in range(steps):
        # Compute pairwise distances efficiently
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Calculate maximum allowable radius for each circle based on current state
        limits_circles = np.min(dists - radii[np.newaxis, :], axis=1)
        limits_bound = np.min(
            np.stack([centers[:, 0], 1 - centers[:, 0], 
                      centers[:, 1], 1 - centers[:, 1]], axis=1), axis=1
        )
        limits = np.minimum(limits_circles, limits_bound)
        
        # Compute repulsive forces for overlaps and boundary violations
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(i+1, N):
                d = dists[i, j]
                r_sum = radii[i] + radii[j]
                if d < r_sum and d > 1e-8:
                    # Stronger repulsion for deeper overlaps
                    repulsion = (r_sum - d) * 10.0
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    fx, fy = repulsion * dx/d, repulsion * dy/d
                    forces[i] += [fx, fy]
                    forces[j] -= [fx, fy]
            
            # Boundary forces
            r = radii[i]
            if centers[i, 0] < r: forces[i, 0] += (r - centers[i, 0]) * 15.0
            if centers[i, 0] > 1-r: forces[i, 0] -= (1-r - centers[i, 0]) * 15.0
            if centers[i, 1] < r: forces[i, 1] += (r - centers[i, 1]) * 15.0
            if centers[i, 1] > 1-r: forces[i, 1] -= (1-r - centers[i, 1]) * 15.0
            
        # Update centers with decaying step size
        current_dt = dt * (0.9992 ** step)
        centers += current_dt * forces
        centers = np.clip(centers, 1e-6, 1-1e-6)
        
        # Grow radii towards the available slack
        # Only increase if there is room, otherwise hold steady while forces resolve conflicts
        radii += 0.08 * np.maximum(0, limits - radii)
        radii = np.clip(radii, 1e-6, 0.5)
        
    # Final strict validation and clamping
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    limits = np.min(np.stack([
        centers[:, 0], 1 - centers[:, 0],
        centers[:, 1], 1 - centers[:, 1],
        np.min(dists - radii[np.newaxis, :], axis=1)
    ], axis=1), axis=1)
    
    radii = np.minimum(radii, limits)
    radii = np.maximum(radii, 1e-6)
    
    return centers, radii, float(np.sum(radii))
