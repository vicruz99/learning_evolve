import numpy as np
import scipy.optimize as opt
import math

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
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
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

def objective_func(params, n_circles):
    """
    Objective function to minimize.
    Minimizes -sum(radii) + penalties for constraint violations.
    params: flattened array [x0, y0, r0, x1, y1, r1, ...]
    """
    centers = params[:2 * n_circles].reshape(n_circles, 2)
    radii = params[2 * n_circles:]
    
    # Penalty for negative radii
    penalty = np.sum(np.maximum(0, -radii))
    
    # Penalty for boundary violations
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => x + r - 1 <= 0
    # Same for y
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        
        # Boundary penalties
        if x - r < 0:
            penalty += 100 * (r - x)**2
        if x + r > 1:
            penalty += 100 * (x + r - 1)**2
        if y - r < 0:
            penalty += 100 * (r - y)**2
        if y + r > 1:
            penalty += 100 * (y + r - 1)**2
            
    # Overlap penalties
    # dist >= r_i + r_j
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap = min_dist - dist
                penalty += 100 * overlap**2
                
    # Objective: maximize sum of radii => minimize -sum(radii)
    obj = -np.sum(radii) + penalty
    return obj

def get_initial_guess_hexagonal(n):
    """
    Generate initial centers and radii based on a hexagonal packing pattern.
    """
    centers = []
    # Try to arrange in rows. 
    # Hexagonal packing: row spacing is r*sqrt(3).
    # Let's assume an initial radius, say 0.1, and place centers.
    # Then the optimizer will adjust.
    
    # Estimate radius for n circles. 
    # Area approx 1. n * pi * r^2 approx 0.9. r approx sqrt(0.9/(n*pi))
    # For n=26, r approx 0.105.
    init_r = 0.105
    
    # Hexagonal grid
    row_height = init_r * math.sqrt(3)
    cols = int(math.ceil(math.sqrt(n * 2 / math.sqrt(3)))) # Rough guess
    # Actually just fill rows
    
    row_idx = 0
    y = init_r
    x_start = init_r
    
    # We can alternate row width or just shift
    shift = 0
    while len(centers) < n:
        # Determine number of circles in this row
        # Max circles in row width 1: floor(1 / (2*init_r)) + 1?
        # With shift, maybe one less.
        
        # Let's try to fit as many as possible in row
        # Width available 1. Diameter 2r.
        # Max count = floor(1 / (2r)) + 1? No, centers at r, 3r, ...
        # Last center at r + (k-1)2r <= 1-r => 2kr <= 1 => k <= 1/(2r)
        # For r=0.105, 1/0.21 approx 4.76 -> 4 circles?
        # But with shift, maybe 5?
        
        # Let's just place circles in a regular hex lattice and stop when we have n.
        
        # Standard hex lattice:
        # Points (i * 2r + (j%2)*r, j * r*sqrt(3)) + offset
        
        # Let's generate points on a grid and select best or just fill
        # Actually simpler: generate a list of candidate positions
        
        pass 

    # Alternative: Just generate a dense grid and pick first n
    # Or better, generate specific rows
    
    # Let's construct rows manually
    # Row 0: 5 circles?
    # Row 1: 4 circles shifted?
    # Row 2: 5 circles?
    # ...
    
    # Let's try a configuration that fits 26 well.
    # 5 rows of 5 is 25. 1 extra.
    # Maybe 5, 5, 5, 5, 4, 2?
    # Or 5, 5, 5, 5, 5, 1?
    
    # Let's try a generic hexagonal filling
    centers = []
    r_est = 0.1
    
    # We will place centers at (x, y)
    # y starts at r_est
    y = r_est
    row_num = 0
    
    while len(centers) < n:
        x = r_est
        shift = (r_est * math.sqrt(3) / 2) # No, horizontal spacing in hex is 2r? 
        # In hex packing, horizontal dist is 2r if aligned, but rows are shifted by r?
        # Actually, centers of touching circles in row are dist 2r apart.
        # Next row is shifted by r horizontally (if circles nestle in gaps).
        # Wait, gap is at x+r. So shift is r.
        
        shift_amt = r_est if row_num % 2 == 1 else 0
        
        # How many circles fit in this row?
        # Start at r_est + shift_amt. End at 1 - r_est + shift_amt?
        # No, center must be >= r_est and <= 1-r_est.
        # With shift, the valid x range is still [r_est, 1-r_est].
        # But the grid points are shifted.
        
        # Let's just place circles with spacing 2*r_est
        # If shifted, first circle might be at r_est + r_est = 2*r_est?
        # No, to be inside, center >= r_est.
        
        # Let's try placing circles at x = r_est + k * 2*r_est + shift_amt
        # We need r_est <= x <= 1-r_est
        
        current_x = r_est + shift_amt
        while current_x <= 1 - r_est + 1e-9:
            centers.append([current_x, y])
            current_x += 2 * r_est
            if len(centers) >= n:
                break
        
        y += r_est * math.sqrt(3)
        row_num += 1
        
    centers = centers[:n]
    
    # Adjust radii to be initial guess
    radii = np.full(n, r_est)
    
    return np.array(centers), radii

def run_packing():
    n_circles = 26
    
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Try multiple random starts or structured starts
    # Structured start 1: Hexagonal
    c1, r1 = get_initial_guess_hexagonal(n_circles)
    # Scale centers to fit better if needed? 
    # The optimizer will handle boundaries, but good initial placement helps.
    # Ensure initial placement is valid-ish (inside bounds)
    c1 = np.clip(c1, r1, 1-r1) # Clamp to avoid huge penalties initially? 
    # Actually clip might move them too much. 
    # Just use as is, optimizer will fix.
    
    initial_params_list = []
    
    # Config 1: Hexagonal
    params1 = np.concatenate([c1.flatten(), r1])
    initial_params_list.append(params1)
    
    # Config 2: Square Grid (5x5 + 1)
    c2 = []
    for i in range(5):
        for j in range(5):
            c2.append([0.1 + i*0.2, 0.1 + j*0.2])
    # Add 26th circle in a gap?
    # Center of square (0.5, 0.5) is occupied.
    # Gap at (0.2, 0.2)? No, occupied.
    # Gap between 4 circles.
    # Maybe place at (0.2, 0.4)? No.
    # Let's place 26th at (0.5, 0.5) with small radius?
    # Or just perturb grid.
    c2.append([0.5, 0.5]) # This will overlap, but optimizer might shrink.
    r2 = np.full(26, 0.1)
    # Shrink the last one to avoid immediate huge penalty
    r2[-1] = 0.01
    params2 = np.concatenate([np.array(c2).flatten(), r2])
    initial_params_list.append(params2)
    
    # Config 3: Random dense packing
    # Place 26 circles randomly, ensure no overlap by sequential placement?
    # Hard to guarantee. Just random.
    np.random.seed(42)
    c3 = np.random.rand(26, 2) * 0.8 + 0.1 # Keep away from edges
    r3 = np.full(26, 0.05)
    params3 = np.concatenate([c3.flatten(), r3])
    initial_params_list.append(params3)
    
    # Config 4: Random with larger radii
    np.random.seed(123)
    c4 = np.random.rand(26, 2) * 0.6 + 0.2
    r4 = np.full(26, 0.08)
    params4 = np.concatenate([c4.flatten(), r4])
    initial_params_list.append(params4)

    # Optimization loop
    for p0 in initial_params_list:
        try:
            # Use L-BFGS-B or SLSQP
            # Bounds: x, y in [0, 1], r >= 0
            bounds = []
            for _ in range(26):
                bounds.append((0, 1)) # x
                bounds.append((0, 1)) # y
                bounds.append((0, 0.5)) # r (max radius 0.5)
            
            result = opt.minimize(
                objective_func,
                p0,
                args=(n_circles,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            if result.success or result.fun < 0: # fun is -sum + penalty
                # Extract results
                res_centers = result.x[:2*n_circles].reshape(n_circles, 2)
                res_radii = result.x[2*n_circles:]
                
                # Check validity manually (penalty might be non-zero but small?)
                # We need strict validity.
                # If penalty is effectively 0, it's good.
                # Recalculate objective without penalty to see true sum?
                # But objective includes penalty.
                # Let's check if it's valid.
                
                # To be safe, we can re-check validity
                if validate_packing(res_centers, res_radii):
                    current_sum = np.sum(res_radii)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = res_centers.copy()
                        best_radii = res_radii.copy()
        except Exception as e:
            print(f"Optimization failed: {e}")

    # If best_sum is still 0 (initial), fallback to a known valid packing
    if best_sum < 1e-5:
        # Fallback to grid
        centers = np.zeros((n_circles, 2))
        radii = np.zeros(n_circles)
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < n_circles:
                    centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                    radii[idx] = 0.1
                    idx += 1
        # Place last one small
        if idx < n_circles:
             # Try to find a spot
             # Just place in center with small radius
             centers[idx] = [0.5, 0.5]
             radii[idx] = 0.01 # Tiny
             idx += 1
        best_centers = centers
        best_radii = radii
        best_sum = np.sum(radii)
        
    # Refinement step: Maybe the optimizer got stuck.
    # Try to increase radii uniformly?
    # Or run another optimization from best_centers
    
    if best_sum > 0:
        # Try to optimize further from best solution
        params_best = np.concatenate([best_centers.flatten(), best_radii])
        try:
            bounds = []
            for _ in range(26):
                bounds.append((0, 1))
                bounds.append((0, 1))
                bounds.append((0, 0.5))
            
            result2 = opt.minimize(
                objective_func,
                params_best,
                args=(n_circles,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 10000}
            )
            
            res_c = result2.x[:2*n_circles].reshape(n_circles, 2)
            res_r = result2.x[2*n_circles:]
            
            if validate_packing(res_c, res_r):
                s = np.sum(res_r)
                if s > best_sum:
                    best_centers = res_c
                    best_radii = res_r
                    best_sum = s
        except:
            pass

    return best_centers, best_radii, best_sum

# To allow running the code
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(c, r)}")
    print(f"Radii: {r}")
    print(f"Centers: {c}")