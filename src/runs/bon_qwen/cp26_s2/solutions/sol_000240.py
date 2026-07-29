# sol_000240 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6efaf445) state=a1448da2 sum of radii=2.496000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    Uses an iterative optimization approach to find the maximum feasible radius
    for 26 equal circles, which serves as a strong baseline for maximizing sum of radii.
    """
    n_circles = 26
    
    # Use a fixed seed for reproducibility
    np.random.seed(42)

    # Initial random centers
    centers = np.random.rand(n_circles, 2)
    
    # Function to compute violation of constraints
    # Violation is sum of squared violations of constraints
    def compute_violation(centers_flat, r):
        centers = centers_flat.reshape(n_circles, 2)
        x = centers[:, 0]
        y = centers[:, 1]
        
        v = 0.0
        
        # Boundary violations
        # Left: x >= r
        diff = r - x
        pos = diff > 0
        if np.any(pos):
            v += np.sum(diff[pos]**2)
        # Right: x <= 1 - r
        diff = x - (1.0 - r)
        pos = diff > 0
        if np.any(pos):
            v += np.sum(diff[pos]**2)
        # Bottom: y >= r
        diff = r - y
        pos = diff > 0
        if np.any(pos):
            v += np.sum(diff[pos]**2)
        # Top: y <= 1 - r
        diff = y - (1.0 - r)
        pos = diff > 0
        if np.any(pos):
            v += np.sum(diff[pos]**2)
            
        # Pairwise violations: dist >= 2r
        # Vectorized distance calculation
        # dx shape (N, N)
        dx = x[:, None] - x
        dy = y[:, None] - y
        dist = np.sqrt(dx**2 + dy**2)
        
        # Upper triangle indices (i < j) to avoid double counting and self
        iu = np.triu_indices(n_circles, k=1)
        pair_dists = dist[iu]
        
        min_dist = 2.0 * r
        diff = min_dist - pair_dists
        pos = diff > 0
        if np.any(pos):
            v += np.sum(diff[pos]**2)
            
        return v

    # Optimize centers for a fixed r
    def optimize_centers(r, init_centers_flat):
        bounds = [(0.0, 1.0)] * (2 * n_circles)
        # Use L-BFGS-B for bounded optimization
        res = minimize(compute_violation, init_centers_flat, args=(r,), 
                       method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12})
        return res.fun, res.x

    # Strategy: 
    # 1. Find a valid configuration for a small r.
    # 2. Iteratively increase r and re-optimize centers.
    
    current_r = 0.05
    current_centers_flat = centers.flatten()
    
    # Initial optimization to ensure validity at starting r
    v, current_centers_flat = optimize_centers(current_r, current_centers_flat)
    best_r = current_r
    best_centers_flat = current_centers_flat
    
    # If initial r=0.05 has violation (unlikely), reduce r until valid
    if v > 1e-9:
        while v > 1e-9 and current_r > 0.01:
            current_r -= 0.005
            v, current_centers_flat = optimize_centers(current_r, current_centers_flat)
        best_r = current_r
        best_centers_flat = current_centers_flat
        
        # Fallback: if still failing, use a perturbed grid
        if v > 1e-9:
            grid_centers = np.zeros((n_circles, 2))
            idx = 0
            for i in range(5):
                for j in range(5):
                    grid_centers[idx] = [0.1 + j*0.2, 0.1 + i*0.2]
                    idx += 1
            # Place 26th circle in a gap (center of square)
            grid_centers[25] = [0.5, 0.5]
            # Perturb to break symmetry/overlap if any
            grid_centers[25] += 0.01 
            best_r = 0.09
            best_centers_flat = grid_centers.flatten()
            v, best_centers_flat = optimize_centers(best_r, best_centers_flat)
            current_r = best_r
            current_centers_flat = best_centers_flat

    # Growth phase: try to increase radius
    step = 0.001
    max_retries = 3 # Number of random restart attempts if stuck
    
    # We try to increase r. 
    for _ in range(1000):
        next_r = current_r + step
        
        # Try to optimize from current centers
        v, new_centers = optimize_centers(next_r, current_centers_flat)
        
        if v < 1e-9:
            current_r = next_r
            current_centers_flat = new_centers
            best_r = current_r
            best_centers_flat = current_centers_flat
        else:
            # Failed to increase r with current centers
            # Try random restarts to escape local minima
            success = False
            for _ in range(max_retries):
                rand_centers = np.random.rand(n_circles, 2)
                v_rand, new_rand = optimize_centers(next_r, rand_centers.flatten())
                if v_rand < 1e-9:
                    current_r = next_r
                    current_centers_flat = new_rand
                    best_r = current_r
                    best_centers_flat = current_centers_flat
                    success = True
                    break
            
            if not success:
                # Could not find valid config for next_r
                break
                
    # Final refinement to clean up numerical errors
    v, final_centers_flat = optimize_centers(best_r, best_centers_flat)
    
    final_centers = final_centers_flat.reshape(n_circles, 2)
    radii = np.full(n_circles, best_r)
    sum_radii = float(np.sum(radii))
    
    return final_centers, radii, sum_radii
