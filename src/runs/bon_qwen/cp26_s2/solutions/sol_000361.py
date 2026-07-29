# sol_000361 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b4024b4) state=3824a25d sum of radii=1.783340 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False

    # Check if radii are nonnegative
    for i in range(n):
        if radii[i] < 0 or np.isnan(radii[i]):
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False

    return True

def compute_score_and_penalty(params, n_circles):
    """
    Computes the sum of radii minus a large penalty for constraint violations.
    params: flat array [x0, y0, r0, x1, y1, r1, ...]
    """
    centers = params[0::3].reshape(n_circles, 2)
    radii = params[1::3].copy() # Skip x, take y? No, structure is x, y, r.
    # Wait, params structure:
    # Index 0: x0, 1: y0, 2: r0
    # Index 3: x1, 4: y1, 5: r1
    
    centers = params.reshape(-1, 3)[:, :2]
    radii = params.reshape(-1, 3)[:, 2]
    
    # Ensure radii are positive for stability, though optimizer might explore negative
    # We clamp radii for penalty calculation but keep params for gradient
    # Actually, better to use transformed variables or hard constraints.
    # Let's just calculate penalty.
    
    score = np.sum(radii)
    penalty = 0.0
    
    # Boundary constraints
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => r + x - 1 <= 0
    # y - r >= 0
    # y + r <= 1
    
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        
        # Violation of x >= r
        viol = max(0, r - x)
        penalty += 1000 * viol**2
        
        # Violation of x + r <= 1
        viol = max(0, r + x - 1)
        penalty += 1000 * viol**2
        
        # Violation of y >= r
        viol = max(0, r - y)
        penalty += 1000 * viol**2
        
        # Violation of y + r <= 1
        viol = max(0, r + y - 1)
        penalty += 1000 * viol**2
        
    # Overlap constraints
    # dist >= r_i + r_j => dist - r_i - r_j >= 0
    # Violation if dist < r_i + r_j
    
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            min_dist = radii[i] + radii[j]
            viol = max(0, min_dist - dist)
            penalty += 1000 * viol**2
            
    return score - penalty

def get_initial_config(n_circles, strategy='hexagonal'):
    """
    Generates an initial configuration of centers and radii.
    """
    centers = np.zeros((n_circles, 2))
    radii = np.ones(n_circles) * 0.05 # Start small
    
    if strategy == 'hexagonal':
        # Try to pack in a hexagonal grid
        # Estimate radius to fit roughly
        # Area ~ 1. 26 * pi * r^2 ~ 1 => r ~ 0.11
        # But we start smaller.
        r_init = 0.08
        spacing = 2 * r_init
        
        # Generate points
        points = []
        row = 0
        col = 0
        # Simple raster scan with offset
        # Max cols approx 1/spacing
        # Max rows approx 1/(spacing * sqrt(3)/2)
        
        # We need 26 points.
        # Let's just fill a grid
        y = r_init
        while len(points) < n_circles:
            x = r_init
            offset = 0
            if row % 2 == 1:
                offset = r_init # Shift by half spacing (which is r)
            
            while x < 1 - r_init and len(points) < n_circles:
                points.append([x, y])
                x += spacing
            row += 1
            y += spacing * np.sqrt(3) / 2
            
        points = np.array(points[:n_circles])
        centers = points
        radii = np.ones(n_circles) * r_init
        
    elif strategy == 'random':
        centers = np.random.rand(n_circles, 2) * 0.6 + 0.2 # Center in square
        radii = np.ones(n_circles) * 0.04
        
    return centers, radii

def run_packing():
    n_circles = 26
    
    best_score = -np.inf
    best_params = None
    
    # We will optimize centers and radii.
    # To make it easier, let's first try to optimize for EQUAL radii.
    # Maximize r subject to constraints.
    # Variables: centers (26, 2) + scalar r.
    # Total vars: 53.
    
    # However, scipy minimize works on 1D arrays.
    
    # Let's try a few initializations and optimize.
    
    strategies = ['hexagonal', 'random', 'random', 'random']
    
    for strat in strategies:
        centers_init, radii_init = get_initial_config(n_circles, strategy=strat)
        
        # Flatten for optimizer
        # Format: x0, y0, r0, x1, y1, r1 ...
        x0 = np.zeros(3 * n_circles)
        for i in range(n_circles):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = radii_init[i]
            
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for i in range(n_circles):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
            
        # Optimization
        # We want to maximize score - penalty.
        # minimize negative score + penalty.
        
        def objective(params):
            return -compute_score_and_penalty(params, n_circles)
            
        # Use L-BFGS-B or SLSQP. SLSQP handles bounds well.
        try:
            res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9})
            
            current_score = -res.fun
            if current_score > best_score:
                # Validate result
                res_centers = res.x.reshape(-1, 3)[:, :2]
                res_radii = res.x.reshape(-1, 3)[:, 2]
                
                # Check validity
                # The penalty function should ensure validity if penalty is high enough
                # But let's double check
                if validate_packing(res_centers, res_radii):
                    best_score = current_score
                    best_params = (res_centers, res_radii)
                else:
                    # If invalid, it means penalty wasn't enough or stuck.
                    # But with 1000*viol^2, it should be very strong.
                    # Maybe just record it if score is high? 
                    # No, must be valid.
                    pass
                    
        except Exception:
            pass

    # If we didn't find a valid one (unlikely with good init), fallback
    if best_params is None:
        # Fallback to simple hexagonal packing with slightly smaller radius
        centers, radii = get_initial_config(n_circles, 'hexagonal')
        # Shrink radii to ensure validity
        # Check overlaps
        valid = False
        shrink_factor = 1.0
        while not valid:
            radii_temp = radii * shrink_factor
            centers_temp = centers
            if validate_packing(centers_temp, radii_temp):
                valid = True
                best_params = (centers_temp, radii_temp)
                best_score = np.sum(radii_temp)
            else:
                shrink_factor *= 0.95
        
    centers_final, radii_final = best_params
    sum_radii = np.sum(radii_final)
    
    # Optional: Local refinement to increase sum of radii with fixed valid topology?
    # Or just return.
    
    # Let's try to improve the sum by a local search on radii if valid.
    # But the optimizer already did that.
    
    return centers_final, radii_final, sum_radii

# To ensure we get a good result, we can increase iterations or runs, 
# but inside the function we are limited by time. 
# The above logic runs 4 optimizations. 
# With N=26, this should be fast enough.
