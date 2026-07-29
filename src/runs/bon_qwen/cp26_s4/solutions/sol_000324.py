# sol_000324 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3035e37b) state=b18e5a1a sum of radii=2.088223 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_radii_grads(centers):
    n = centers.shape[0]
    # Pairwise differences and distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_dists = np.min(dists, axis=1)
    min_idx = np.argmin(dists, axis=1)
    
    # Boundary distances: [x, 1-x, y, 1-y]
    b_dists = np.array([
        centers[:, 0],
        1 - centers[:, 0],
        centers[:, 1],
        1 - centers[:, 1]
    ]).T
    
    b_mins = np.min(b_dists, axis=1)
    b_min_idx = np.argmin(b_dists, axis=1)
    
    radii = np.zeros(n)
    grads = np.zeros_like(centers)
    
    for i in range(n):
        # Determine if circle-circle or circle-boundary constraint is tighter
        if min_dists[i] < b_mins[i]:
            radii[i] = min_dists[i] * 0.5
            # Gradient points away from the limiting circle
            dir = diffs[i, min_idx[i]] / np.maximum(min_dists[i], 1e-12)
            grads[i] = 0.5 * dir
        else:
            radii[i] = b_mins[i] * 0.5
            # Gradient points inward from the limiting boundary
            idx = b_min_idx[i]
            if idx == 0: grads[i, 0] = 0.5
            elif idx == 1: grads[i, 0] = -0.5
            elif idx == 2: grads[i, 1] = 0.5
            else: grads[i, 1] = -0.5
            
    return radii, grads

def run_packing():
    n = 26
    best_sum_r = 0.0
    best_centers = np.random.rand(n, 2) * 0.8 + 0.1
    
    for restart in range(15):
        if restart == 0:
            # Hexagonal lattice initialization for strong starting point
            pts = []
            for row in range(6):
                y = row * 0.17 + 0.12
                cols = 5 if row % 2 == 0 else 4
                for col in range(cols):
                    x = col * 0.21 + (0.1 if row % 2 != 0 else 0.05)
                    if len(pts) < n:
                        pts.append([x, y])
            centers = np.array(pts[:n])
        else:
            # Random initialization with jitter
            centers = np.random.rand(n, 2) * 0.8 + 0.1
            
        centers += np.random.randn(n, 2) * 0.002
        centers = np.clip(centers, 0.02, 0.98)
        
        current_centers = centers.copy()
        step = 0.04
        
        for it in range(4000):
            radii, grads = compute_radii_grads(current_centers)
            current_sum = np.sum(radii)
            
            if current_sum > best_sum_r:
                best_sum_r = current_sum
                best_centers = current_centers.copy()
                
            # Gradient ascent step
            current_centers += step * grads
            current_centers = np.clip(current_centers, 0.01, 0.99)
            
            # Exponential decay of step size for convergence
            step *= 0.9995
            
            if step < 1e-8:
                break
                
    # Final radii computation
    radii, _ = compute_radii_grads(best_centers)
    
    # Slight shrinkage to guarantee validation passes with 1e-12 tolerance
    radii *= 0.99999
    radii = np.maximum(radii, 1e-9)
    
    return best_centers, radii, np.sum(radii)
