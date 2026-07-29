# sol_000230 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b088ff81) state=85b9fe41 sum of radii=2.132481 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def get_hex_initial_guess(n, r_guess):
    """Generates a hexagonal lattice initial guess for n circles."""
    rows = []
    # Estimate rows and columns for hex packing
    r = r_guess
    dx = 2.0 * r
    dy = math.sqrt(3.0) * r
    
    y = r
    count = 0
    centers = []
    
    # Generate a grid
    while count < n:
        x = r
        # Determine how many fit in width 1
        # Width required for m circles is 2*m*r
        max_circles = int(1.0 / (2.0 * r))
        if max_circles < 1:
            max_circles = 1 # Fallback
            
        needed = n - count
        m = min(max_circles, needed)
        
        for _ in range(m):
            centers.append([x, y])
            x += dx
            count += 1
            if count >= n:
                break
        if count >= n:
            break
        y += dy
        
    return np.array(centers)

def loss_function_equal_r(params, n):
    """
    Objective function to minimize.
    Minimizes -r + penalty(overlap, boundaries).
    Encourages large r while keeping configuration valid.
    """
    coords = params[:2*n].reshape(n, 2)
    r = params[2*n]
    
    if r < 1e-6:
        return 1e6
    
    penalty = 0.0
    lambda_penalty = 100.0 # Weight for constraints
    
    # Boundary constraints
    for i in range(n):
        x, y = coords[i]
        # Left
        val = x - r
        if val < 0: penalty += val**2
        # Right
        val = 1.0 - (x + r)
        if val < 0: penalty += val**2
        # Bottom
        val = y - r
        if val < 0: penalty += val**2
        # Top
        val = 1.0 - (y + r)
        if val < 0: penalty += val**2
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = math.sqrt((coords[i,0] - coords[j,0])**2 + (coords[i,1] - coords[j,1])**2)
            min_dist = 2.0 * r
            if dist < min_dist:
                penalty += (min_dist - dist)**2
                
    return -r + lambda_penalty * penalty

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    
    # Strategy:
    # 1. Generate a good initial configuration (hexagonal grid).
    # 2. Use Nelder-Mead to optimize the radius and positions for equal circles.
    # 3. Return the result.
    
    # Initial guess parameters
    # Try a radius slightly below the theoretical limit for 26 circles
    # Theoretical limit for equal circles in square is around 0.101-0.105
    r_start = 0.095 
    
    # Generate initial centers
    centers_init = get_hex_initial_guess(n, r_start)
    
    # If the guess didn't produce n centers (unlikely with small r), pad or adjust
    if centers_init.shape[0] < n:
        # Fallback to grid
        step = 0.2
        centers_list = []
        for i in range(5):
            for j in range(5):
                centers_list.append([0.1 + i*step, 0.1 + j*step])
        centers_list.append([0.5, 0.95])
        centers_init = np.array(centers_list[:n])

    # Initial parameters vector: [x1, y1, ..., xn, yn, r]
    params0 = np.concatenate([centers_init.flatten(), [r_start]])
    
    # Optimization using Nelder-Mead (derivative-free, good for non-smooth)
    res = opt.minimize(lambda p: loss_function_equal_r(p, n), params0, method='Nelder-Mead', 
                       options={'maxiter': 10000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
    
    r_opt = res.x[-1]
    coords_opt = res.x[:2*n].reshape(n, 2)
    
    # Validation and safety check
    # If penalty was not zero, the solution might be slightly invalid.
    # We can check validity and scale down r if needed.
    
    def check_validity(coords, r):
        # Check boundaries
        if np.any(coords[:, 0] - r < -1e-7) or np.any(coords[:, 0] + r > 1 + 1e-7):
            return False
        if np.any(coords[:, 1] - r < -1e-7) or np.any(coords[:, 1] + r > 1 + 1e-7):
            return False
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < 2 * r - 1e-7:
                    return False
        return True

    if not check_validity(coords_opt, r_opt):
        # Reduce radius slightly to ensure validity
        # Heuristic reduction
        r_opt *= 0.98
        # Re-check or just trust reduction
        # Better: find max valid r by scaling
        # But simple reduction is usually safe for small violations
        
        # Re-validate after reduction
        if not check_validity(coords_opt, r_opt):
            r_opt *= 0.95 
            
    radii_opt = np.full(n, r_opt)
    sum_radii = np.sum(radii_opt)
    
    return coords_opt, radii_opt, sum_radii
