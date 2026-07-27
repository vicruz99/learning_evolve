# sol_000063 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cae61cda) state=96e87d9b sum of radii=1.711378 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the positions and radii of 26 circles in a unit square 
    to maximize the sum of radii using Simulated Annealing.
    """
    np.random.seed(42)
    N = 26
    
    # 1. Initialization: Grid layout scaled to fit inside [0,1]^2
    centers = np.zeros((N, 2))
    idx = 0
    for i in range(5):
        for j in range(6):
            if idx < N:
                centers[idx] = [0.08 + j * 0.175, 0.08 + i * 0.175]
                idx += 1
                
    # Initial radii small enough to avoid immediate overlap
    radii = np.full(N, 0.045)
    
    current_sum = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = current_sum
    
    # SA Parameters
    T = 0.6
    T_min = 1e-8
    alpha = 0.9996
    step_init = 0.035
    
    # Preallocate temporary arrays for performance
    diff = np.zeros((N, 2))
    
    iter_count = 0
    max_iter = 1500000
    
    while iter_count < max_iter and T > T_min:
        # Periodic global perturbation to escape deep local minima
        if iter_count > 0 and iter_count % 60000 == 0:
            centers += np.random.uniform(-0.015, 0.015, (N, 2))
            centers = np.clip(centers, 0.02, 0.98)
            # Recompute radii to restore feasibility after shake
            for k in range(N):
                nc = centers[k]
                r_bound = min(nc[0], 1.0 - nc[0], nc[1], 1.0 - nc[1])
                diff[:] = centers - nc
                diff[k] = 0.0
                dists = np.sqrt(np.sum(diff**2, axis=1))
                limits = dists - radii
                limits[k] = r_bound
                radii[k] = max(1e-4, np.min(limits))
            current_sum = np.sum(radii)

        # Select a random circle to move
        i = np.random.randint(N)
        
        # Propose new center
        step = step_init * T
        dc = np.random.uniform(-step, step, 2)
        nc = centers[i] + dc
        nc = np.clip(nc, 0.005, 0.995)  # Keep strictly inside bounds
        
        # Compute distances to all other circles
        diff[:] = centers - nc
        diff[i] = 0.0
        dists = np.sqrt(np.sum(diff**2, axis=1))
        
        # Constraint 1: Boundary
        r_bound = min(nc[0], 1.0 - nc[0], nc[1], 1.0 - nc[1])
        
        # Constraint 2: Non-overlap with neighbors
        # r_i + r_j <= dist_ij  =>  r_i <= dist_ij - r_j
        limits = dists - radii
        limits[i] = r_bound  # Apply self-boundary constraint
        r_max = np.min(limits)
        
        # If position is completely infeasible, reject move
        if r_max < 1e-6:
            T *= alpha
            iter_count += 1
            continue
            
        # Propose new radius: usually maximal, occasionally smaller to explore
        if np.random.rand() < 0.03:
            nr = np.random.uniform(1e-4, r_max)
        else:
            nr = r_max
            
        delta = nr - radii[i]
        
        # Acceptance criterion
        if delta >= 0 or np.random.rand() < math.exp(delta / T):
            centers[i] = nc
            radii[i] = nr
            current_sum += delta
            
            # Update best solution
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
                
        T *= alpha
        iter_count += 1
        
    return best_centers, best_radii, float(best_sum)
