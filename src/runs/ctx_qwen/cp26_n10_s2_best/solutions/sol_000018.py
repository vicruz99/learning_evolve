# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b9ac6cc) state=7bf3fd68 sum of radii=1.738456 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def get_max_radii(centers):
    """
    Given centers, compute the maximum possible radius for each circle
    such that they don't overlap and stay inside [0,1]x[0,1].
    """
    n = centers.shape[0]
    radii = np.zeros(n)
    
    # Precompute distances to walls
    # distance to left (x=0) is x, to right (x=1) is 1-x, etc.
    dists_to_walls = np.minimum(
        np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
        np.minimum(centers[:, 1], 1.0 - centers[:, 1])
    )
    
    # Distance to other circles
    # dist[i] will store min distance to any other circle center
    # We want min( dist_to_wall, 0.5 * min_dist_to_neighbor )
    
    # Compute pairwise distances
    # diff[i, j] = centers[i] - centers[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Set diagonal to infinity so we don't consider self
    np.fill_diagonal(dists, np.inf)
    
    min_dist_to_neighbor = np.min(dists, axis=1)
    
    # The radius is limited by the closest obstacle (wall or neighbor)
    # If neighbor is closer (scaled by 0.5), it limits radius.
    radii = np.minimum(dists_to_walls, 0.5 * min_dist_to_neighbor)
    
    return radii

def objective(centers_flat):
    """
    Objective function for optimization.
    Minimizes the negative sum of radii.
    """
    n = 26
    centers = centers_flat.reshape(n, 2)
    
    # Keep centers strictly inside (0,1) to avoid numerical issues with min function
    # Although get_max_radii handles it, optimization might push to boundaries
    centers = np.clip(centers, 1e-5, 1.0 - 1e-5)
    
    radii = get_max_radii(centers)
    return -np.sum(radii)

def run_packing():
    n = 26
    best_sum = -np.inf
    best_centers = None
    
    # We run the optimizer multiple times with different initial configurations
    # to find a better global optimum.
    
    for _ in range(10):
        # Initial guess: Hexagonal-like grid
        # 26 circles. sqrt(26) is approx 5.1.
        # We can arrange in 5 rows.
        # Row counts: 6, 5, 5, 5, 5 (Total 26) or similar.
        # Let's create a grid and perturb it.
        
        cols = 6
        rows = 5
        # This fits 30 spots, we need 26.
        # Let's just place them on a grid and remove 4? 
        # Or just random initialization within bounds is safer for hill climbing.
        
        # Deterministic initialization based on grid
        centers_init = np.zeros((n, 2))
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx < n:
                    # Hexagonal offset
                    x_offset = (r % 2) * 0.5 * (1.0 / cols) 
                    # Spacing
                    dx = 1.0 / (cols - 1) if cols > 1 else 0.5
                    dy = 1.0 / (rows - 1) if rows > 1 else 0.5
                    
                    x = c * dx + x_offset
                    y = r * dy
                    
                    # Add small random noise
                    x += np.random.uniform(-0.02, 0.02)
                    y += np.random.uniform(-0.02, 0.02)
                    
                    centers_init[idx] = [x, y]
                    idx += 1
        
        # Flatten centers for optimizer
        x0 = centers_init.flatten()
        
        # Optimize
        # Nelder-Mead is good for non-smooth functions
        res = opt.minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-7})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = res.x.reshape(n, 2).copy()
            
    # Final validation and radius calculation
    # Clip centers to ensure they are valid
    best_centers = np.clip(best_centers, 0.0, 1.0)
    
    # Compute final valid radii
    radii = get_max_radii(best_centers)
    
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    return best_centers, radii, np.sum(radii)
