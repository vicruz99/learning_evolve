# sol_000013 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state accdaaf6) state=420b925e sum of radii=1.321801 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_min_distance(centers, n):
    """Compute the minimum distance between any pair of centers or to boundaries."""
    min_d = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = centers[j] - centers[i]
            d = np.sqrt(np.sum(diff**2))
            if d < min_d:
                min_d = d
        # Distance to square boundaries [0,1]x[0,1]
        min_d = min(min_d, centers[i, 0], 1.0 - centers[i, 0], 
                          centers[i, 1], 1.0 - centers[i, 1])
    return min_d

def neg_min_dist_objective(x, n):
    """Objective function for scipy: minimize negative minimum distance."""
    centers = x.reshape(n, 2)
    return -get_min_distance(centers, n)

def run_packing():
    n = 26
    np.random.seed(42)
    
    # 1. Initialize with a perturbed hexagonal lattice
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(6):
        for col in range(5):
            x = 0.08 + col * 0.18
            y = 0.08 + row * 0.16
            if row % 2 == 1:
                x += 0.09
            if idx < n and 0 <= x <= 1 and 0 <= y <= 1:
                centers[idx] = [x + np.random.randn() * 0.005, y + np.random.randn() * 0.005]
                idx += 1
                
    # 2. Force-directed relaxation with gradually increasing target radius
    r = 0.06
    for step in range(3000):
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[j] - centers[i]
                dist = np.sqrt(np.sum(diff**2))
                if dist < 2 * r and dist > 1e-12:
                    rep = (2 * r - dist) / dist
                    forces[i] += diff * rep
                    forces[j] -= diff * rep
            # Boundary repulsion
            if centers[i, 0] < r: forces[i, 0] += r - centers[i, 0]
            if centers[i, 0] > 1 - r: forces[i, 0] -= centers[i, 0] - (1 - r)
            if centers[i, 1] < r: forces[i, 1] += r - centers[i, 1]
            if centers[i, 1] > 1 - r: forces[i, 1] -= centers[i, 1] - (1 - r)
            
        # Adaptive step size with cooling
        step_size = 0.004 * (1.0 - step / 3000.0) + 0.0001
        centers += step_size * forces
        centers = np.clip(centers, 1e-8, 1 - 1e-8)
        
        # Slowly increase target radius to pack denser
        if step % 30 == 0 and r < 0.102:
            r += 0.0003
            
    # 3. Fine-tune positions using Nelder-Mead to maximize minimum clearance
    x0 = centers.flatten()
    res = minimize(neg_min_dist_objective, x0, args=(n,), method='Nelder-Mead',
                   options={'maxiter': 8000, 'xatol': 1e-8, 'fatol': 1e-8})
                   
    best_centers = res.x.reshape(n, 2)
    # Enforce strict bounds to avoid numerical edge cases
    best_centers = np.clip(best_centers, 1e-9, 1 - 1e-9)
    
    # 4. Compute optimal equal radius and results
    min_dist = get_min_distance(best_centers, n)
    final_r = min_dist / 2.0
    radii = np.full(n, final_r)
    sum_radii = float(np.sum(radii))
    
    return best_centers, radii, sum_radii
