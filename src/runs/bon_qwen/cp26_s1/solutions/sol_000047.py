# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ff99986a) state=650b0204 sum of radii=2.540000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

N_CIRCLES = 26
PENALTY_SCALE = 1000.0

def check_overlap_and_boundary(centers, radii):
    """
    Calculates a penalty score for the configuration.
    Returns (sum_radii, penalty).
    """
    n = centers.shape[0]
    sum_r = np.sum(radii)
    penalty = 0.0
    
    # Boundary checks
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        
        if x < r:
            penalty += (r - x)**2 * PENALTY_SCALE
        elif x > 1 - r:
            penalty += (x - (1 - r))**2 * PENALTY_SCALE
            
        if y < r:
            penalty += (r - y)**2 * PENALTY_SCALE
        elif y > 1 - r:
            penalty += (y - (1 - r))**2 * PENALTY_SCALE
            
    # Overlap checks
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.0
            
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                penalty += (min_dist - dist)**2 * PENALTY_SCALE
                
    return sum_r, penalty

def objective_function_26(vars):
    """
    Objective function for scipy optimization.
    Minimizes -sum_radii + penalty.
    """
    n = 26
    centers = vars.reshape(-1, 3)[:, :2]
    radii = vars.reshape(-1, 3)[:, 2]
    sum_r, penalty = check_overlap_and_boundary(centers, radii)
    return -sum_r + penalty

def get_grid_init():
    """
    5x5 grid + 1 circle in gap.
    """
    centers = []
    radii = []
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            centers.append([0.1 + i*0.2, 0.1 + j*0.2])
            radii.append(0.1)
    # 1 in gap
    centers.append([0.2, 0.2])
    radii.append(0.04)
    return np.array(centers), np.array(radii)

def get_random_init():
    """
    Random initialization.
    """
    centers = np.random.rand(26, 2)
    radii = np.random.rand(26) * 0.05 + 0.01
    return centers, radii

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Initialize with grid and random starts
    inits = [get_grid_init(), get_random_init(), get_random_init(), get_random_init()]
    
    # Bounds for optimization: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (3 * N_CIRCLES)
    for i in range(0, 3*N_CIRCLES, 3):
        bounds[i+2] = (0.0, 0.5)
        
    for c, r in inits:
        # Add small noise to escape local minima
        c += np.random.normal(0, 0.001, c.shape)
        r += np.random.normal(0, 0.001, r.shape)
        r = np.maximum(r, 0.001)
        
        # Flatten to 1D array for optimizer
        x0 = []
        for i in range(N_CIRCLES):
            x0.extend([c[i, 0], c[i, 1], r[i]])
            
        # Optimize
        res = opt.minimize(objective_function_26, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 5000, 'ftol': 1e-12})
        
        f_centers = res.x.reshape(-1, 3)[:, :2]
        f_radii = res.x.reshape(-1, 3)[:, 2]
        
        # Check validity
        s, p = check_overlap_and_boundary(f_centers, f_radii)
        if p < 1e-5 and s > best_sum:
            best_sum = s
            best_centers = f_centers
            best_radii = f_radii
            
    if best_centers is None:
        best_centers, best_radii = get_grid_init()
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum
