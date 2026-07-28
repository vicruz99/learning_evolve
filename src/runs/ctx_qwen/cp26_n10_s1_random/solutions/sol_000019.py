# sol_000019 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a75b8609) state=79dfa8a0 sum of radii=1.445107 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def objective_function(vars, n):
    """
    Objective function for optimization.
    Maximizes sum of radii (minimizes negative sum) plus penalty for constraint violations.
    """
    centers_opt = vars[:2*n].reshape(n, 2)
    radii_opt = vars[2*n:]
    
    w_boundary = 5000.0
    w_overlap = 10000.0
    penalty = 0.0
    
    # Boundary penalties
    for i in range(n):
        x, y = centers_opt[i]
        r = radii_opt[i]
        
        if x < r: penalty += (x - r)**2 * w_boundary
        if x > 1 - r: penalty += (x - (1 - r))**2 * w_boundary
        if y < r: penalty += (y - r)**2 * w_boundary
        if y > 1 - r: penalty += (y - (1 - r))**2 * w_boundary
        if r < 0: penalty += r**2 * w_boundary
        
    # Overlap penalties
    for i in range(n):
        xi, yi = centers_opt[i]
        ri = radii_opt[i]
        for j in range(i + 1, n):
            xj, yj = centers_opt[j]
            rj = radii_opt[j]
            
            min_d = ri + rj
            min_d_sq = min_d * min_d
            
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            
            if dist_sq < min_d_sq:
                dist = math.sqrt(dist_sq)
                overlap = min_d - dist
                penalty += overlap**2 * w_overlap
                
    return -np.sum(radii_opt) + penalty

def is_valid_check(centers, radii, n):
    """
    Checks if the packing is valid according to strict rules.
    """
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or \
           y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
        for j in range(i + 1, n):
            dist = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

def run_packing():
    n = 26
    
    # 1. Initialization
    np.random.seed(123)
    centers = []
    # 5x5 grid for 25 circles
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    # 26th circle in a gap
    centers.append([0.2, 0.2])
    centers = np.array(centers)
    
    # Perturb to break symmetry
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    radii = np.ones(n) * 0.04
    
    # 2. Pre-optimization: Force-directed spreading
    # Try to increase radii and resolve overlaps iteratively
    for step in range(100):
        target_r = 0.04 + step * 0.001
        
        # Resolve overlaps for the current target radius
        for _ in range(20):
            for i in range(n):
                xi, yi = centers[i]
                for j in range(i + 1, n):
                    xj, yj = centers[j]
                    
                    dx = xi - xj
                    dy = yi - yj
                    dist_sq = dx*dx + dy*dy
                    min_dist = 2 * target_r
                    min_dist_sq = min_dist * min_dist
                    
                    if dist_sq < min_dist_sq and dist_sq > 1e-12:
                        dist = math.sqrt(dist_sq)
                        overlap = min_dist - dist
                        # Push apart
                        shift = overlap * 0.1
                        nx, ny = dx/dist, dy/dist
                        centers[i, 0] -= nx * shift
                        centers[i, 1] -= ny * shift
                        centers[j, 0] += nx * shift
                        centers[j, 1] += ny * shift
            
            # Enforce boundary constraints relative to target_r
            for i in range(n):
                xi, yi = centers[i]
                centers[i, 0] = np.clip(xi, target_r, 1.0 - target_r)
                centers[i, 1] = np.clip(yi, target_r, 1.0 - target_r)
        
        # Estimate actual feasible radius based on current spacing
        min_d = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < min_d: min_d = d
            # Check boundaries
            min_d = min(min_d, centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        
        radii = np.ones(n) * (min_d / 2.0)
    
    # 3. Optimization
    x0 = np.concatenate([centers.flatten(), radii])
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # Use L-BFGS-B with penalty objective
    res = minimize(objective_function, x0, args=(n,), method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 3000, 'ftol': 1e-12})
    
    final_vars = res.x
    final_centers = final_vars[:2*n].reshape(n, 2)
    final_radii = final_vars[2*n:]
    
    # 4. Post-processing to ensure validity
    if not is_valid_check(final_centers, final_radii, n):
        factor = 1.0
        while not is_valid_check(final_centers, final_radii, n):
            factor *= 0.995
            final_radii *= factor
            if np.sum(final_radii) < 1.0: break 
            
    return final_centers, final_radii, np.sum(final_radii)
