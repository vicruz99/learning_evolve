# sol_000012 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state abc5794a) state=0433a958 sum of radii=0.001300 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def get_overlap_penalty(centers, radii):
    """
    Calculates a penalty based on overlaps between circles.
    Penalty is sum of squared overlap distances.
    """
    n = centers.shape[0]
    penalty = 0.0
    
    # Vectorized distance calculation
    # diff: (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # radii_sum: (n, n)
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Overlap amount: positive if overlap
    overlaps = radii_sum - dists
    
    # We only care about i < j, but squaring the whole matrix and dividing by 2 is easier
    # or just take upper triangle. 
    # Overlap is 2 * sum_{i<j} max(0, r_i + r_j - d_ij)^2
    # Here we compute for all pairs and divide by 2.
    
    overlaps[overlaps < 0] = 0.0
    penalty = np.sum(overlaps**2) * 0.5 # 0.5 to correct for double counting
    
    return penalty

def get_boundary_penalty(centers, radii):
    """
    Calculates penalty for circles going outside [0, 1]x[0, 1].
    """
    # x - r < 0  => r - x > 0
    # x + r > 1  => x + r - 1 > 0
    # y - r < 0
    # y + r > 1
    
    violations = np.zeros_like(radii)
    
    # Left/Bottom boundaries
    # x - r >= 0  => r <= x. Violation if r > x. Amount: r - x
    v1 = radii - centers[:, 0]
    v1[v1 < 0] = 0.0
    
    # Right/Top boundaries
    # 1 - x - r >= 0 => r <= 1 - x. Violation if r > 1 - x. Amount: r - (1 - x)
    v2 = radii - (1.0 - centers[:, 0])
    v2[v2 < 0] = 0.0
    
    v3 = radii - centers[:, 1]
    v3[v3 < 0] = 0.0
    
    v4 = radii - (1.0 - centers[:, 1])
    v4[v4 < 0] = 0.0
    
    penalty = np.sum(v1**2) + np.sum(v2**2) + np.sum(v3**2) + np.sum(v4**2)
    return penalty

def objective_function(vars, n_circles):
    """
    Objective function to minimize.
    vars = [x1, y1, r1, x2, y2, r2, ...]
    We want to maximize sum(r), so minimize -sum(r).
    Plus penalties.
    """
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        idx = i * 3
        centers[i, 0] = vars[idx]
        centers[i, 1] = vars[idx+1]
        radii[i] = max(0.0, vars[idx+2]) # Radii must be non-negative
    
    # We want to maximize sum(radii) => minimize -sum(radii)
    obj = -np.sum(radii)
    
    # Penalty weights. High weights ensure constraints are respected.
    # However, if weights are too high, gradient descent might get stuck or be ill-conditioned.
    # A large weight like 1000 or 10000 is typical.
    penalty_weight = 5000.0
    
    obj += penalty_weight * get_overlap_penalty(centers, radii)
    obj += penalty_weight * get_boundary_penalty(centers, radii)
    
    return obj

def generate_hexagonal_init(n, r_guess):
    """
    Generates initial centers for n circles in a hexagonal pattern.
    """
    # Estimate dimensions
    # Area approx n * 2*sqrt(3)*r^2 ~ 1 => r ~ sqrt(1 / (n * 2*sqrt(3)))
    # But we use r_guess.
    
    centers = []
    
    # Try to fit in a grid
    # Rows
    num_rows = int(np.ceil(np.sqrt(n * 2 / math.sqrt(3))))
    
    # Rough count per row
    # Let's just place them and see.
    
    idx = 0
    row = 0
    while idx < n:
        y = row * math.sqrt(3) * r_guess
        x_offset = (row % 2) * r_guess
        
        # How many fit in this row?
        # Width available 1.0. 
        # If we assume r_guess fits, width used ~ 2*r_guess * count
        # But let's just place them.
        
        col = 0
        while idx < n:
            x = x_offset + col * 2 * r_guess
            # Check if fits in [0, 1] roughly
            if x + r_guess > 1.0:
                break
            centers.append([x + r_guess, y + r_guess]) # Centered at r from origin?
            # Actually, better to center in [0,1] later. 
            # Let's just place centers at x, y.
            centers[-1][0] = x + r_guess # Shift to be safe? No.
            centers[-1][1] = y + r_guess
            
            # Correct placement: center at (x+r, y+r) if x,y is top-left of bounding box?
            # Let's just place center at x, y.
            # To fit in [0,1], center must be in [r, 1-r].
            # So x must be in [r, 1-r].
            
            # Let's restart logic for cleaner generation.
            pass
            col += 1
            idx += 1
        row += 1

    # Better generation:
    centers = []
    y = r_guess
    row_idx = 0
    while len(centers) < n:
        x = r_guess
        shift = r_guess if row_idx % 2 == 1 else 0
        x += shift
        while x + r_guess <= 1.0 + 1e-9:
            if len(centers) < n:
                centers.append([x, y])
            x += 2 * r_guess
        y += math.sqrt(3) * r_guess
        row_idx += 1
        
    # If we have fewer than n, add random or duplicates (optimizer will move them)
    while len(centers) < n:
        centers.append([0.5, 0.5])
        
    # Normalize to fit in [0, 1] roughly?
    # The generation logic puts centers in [r, 1-r] roughly.
    # But if r_guess is small, they are in [0, 1].
    
    return np.array(centers[:n])

def optimize_packing():
    n = 26
    
    # Best result tracking
    best_vars = None
    best_score = -np.inf # We want to maximize sum radii, so store max sum
    best_centers = None
    best_radii = None
    
    # Try multiple initial guesses
    # Strategy 1: Hexagonal grid with varying radii
    # Strategy 2: Random perturbations
    
    attempts = 20
    
    for attempt in range(attempts):
        # Randomize r_guess slightly around expected optimal ~0.105
        # Target sum 2.636 => avg r ~ 0.1014.
        # Let's start with 0.10
        r_init = 0.10 + np.random.uniform(-0.01, 0.02)
        
        # Generate centers
        centers_init = generate_hexagonal_init(n, r_init)
        
        # Random perturbation
        centers_init += np.random.uniform(-0.02, 0.02, centers_init.shape)
        
        # Clip to [0.05, 0.95] to avoid immediate boundary issues
        centers_init = np.clip(centers_init, 0.05, 0.95)
        
        # Initialize radii
        radii_init = np.full(n, r_init)
        
        # Flatten variables
        initial_vars = []
        for i in range(n):
            initial_vars.extend([centers_init[i, 0], centers_init[i, 1], radii_init[i]])
        
        initial_vars = np.array(initial_vars)
        
        # Bounds
        bounds = []
        for i in range(n):
            bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)]) # x, y, r
            
        # Optimization
        # Using L-BFGS-B with bounds
        try:
            res = minimize(
                objective_function, 
                initial_vars, 
                args=(n,), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-9}
            )
            
            # Extract result
            if res.success or res.fun < 0: # Just check if we got a result
                centers_opt = np.zeros((n, 2))
                radii_opt = np.zeros(n)
                for i in range(n):
                    idx = i * 3
                    centers_opt[i] = res.x[idx : idx+2]
                    radii_opt[i] = res.x[idx+2]
                
                # Calculate sum radii
                current_sum = np.sum(radii_opt)
                
                # Validation check (strict)
                # We need to ensure no overlap. 
                # The penalty method might leave tiny overlaps if not fully converged.
                # Let's do a quick check.
                valid = True
                for i in range(n):
                    x, y = centers_opt[i]
                    r = radii_opt[i]
                    if x < 0 or x > 1 or y < 0 or y > 1 or r < 0:
                        valid = False
                        break
                    for j in range(i+1, n):
                        dist = np.sqrt((centers_opt[i,0]-centers_opt[j,0])**2 + (centers_opt[i,1]-centers_opt[j,1])**2)
                        if dist < radii_opt[i] + radii_opt[j] - 1e-7:
                            valid = False
                            break
                    if not valid: break
                
                if valid and current_sum > best_score:
                    best_score = current_sum
                    best_centers = centers_opt.copy()
                    best_radii = radii_opt.copy()
                    
        except Exception as e:
            continue

    # If we didn't find a good valid packing, fallback to a safe one
    if best_centers is None:
        # Fallback: 5x5 grid plus some random small circles?
        # Or just 25 circles in grid and 1 tiny?
        # But we need 26.
        # Let's construct a valid grid packing.
        r = 0.1
        centers = np.zeros((26, 2))
        radii = np.full(26, r)
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < 26:
                    centers[idx] = [0.1 + j*0.2, 0.1 + i*0.2]
                    idx += 1
        # Place 26th circle? 
        # In 5x5 grid of r=0.1, there is no space.
        # We must reduce radius.
        # Let's just use the best result from optimization, even if slightly invalid?
        # No, must be valid.
        # If optimization failed, let's try to shrink radii of the best result until valid.
        if best_centers is None:
             # Fallback to equal circles optimization with strict projection?
             # Just return a valid but suboptimal packing.
             # 5x5 grid with r=0.1 fits 25. 
             # To fit 26, maybe r=0.09?
             r = 0.09
             centers = np.zeros((26, 2))
             radii = np.full(26, r)
             idx = 0
             # 6 rows? 5, 5, 5, 5, 5, 1?
             # Height 0.1 + 4*0.18 + 0.1?
             # Let's just do 5x5 and put one in center? No overlap.
             # Let's do 4 rows of 5 (20) + 3 rows of 2?
             # Just a valid packing.
             # 5x5 grid r=0.1 is valid for 25.
             # To fit 26, reduce r.
             # Try r=0.095.
             r = 0.095
             centers = np.zeros((26, 2))
             radii = np.full(26, r)
             idx = 0
             # 5 rows of 5 = 25.
             # 6th circle?
             # Maybe 5, 5, 5, 5, 5, 1 arrangement?
             # Or 5, 5, 5, 5, 4, 2?
             # Let's try to place 26 circles with r=0.095 in hex pattern.
             # Hex pattern logic
             centers_list = []
             y = r
             row = 0
             while len(centers_list) < 26:
                 x = r
                 shift = r if row % 2 == 1 else 0
                 x += shift
                 while x + r <= 1.0 + 1e-9:
                     centers_list.append([x, y])
                     x += 2 * r
                 y += math.sqrt(3) * r
                 row += 1
             
             best_centers = np.array(centers_list[:26])
             best_radii = np.full(26, r)
             best_score = np.sum(best_radii)

    # Final validation and correction
    # If best solution has tiny overlaps, shrink radii slightly.
    # This ensures validity.
    # Check overlaps
    max_overlap = 0.0
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
            overlap = (best_radii[i] + best_radii[j]) - dist
            if overlap > max_overlap:
                max_overlap = overlap
        
        # Boundary
        x, y = best_centers[i]
        r = best_radii[i]
        if x - r < 0: max_overlap = max(max_overlap, -(x-r))
        if x + r > 1: max_overlap = max(max_overlap, (x+r)-1)
        if y - r < 0: max_overlap = max(max_overlap, -(y-r))
        if y + r > 1: max_overlap = max(max_overlap, (y+r)-1)

    if max_overlap > 1e-12:
        # Shrink all radii by half the max overlap to be safe
        # Or just scale down slightly
        # Uniform shrinkage
        shrink = max_overlap / 2.0 + 1e-6
        best_radii -= shrink
        best_radii = np.maximum(best_radii, 0.0)

    return best_centers, best_radii, float(np.sum(best_radii))

def run_packing():
    centers, radii, sum_radii = optimize_packing()
    return centers, radii, sum_radii
