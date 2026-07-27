import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal lattice initialization and local optimization.
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal Grid
    # We need to distribute 26 circles in a hexagonal pattern.
    # A common pattern is rows with alternating number of circles.
    # Let's try to fit 26 circles.
    # Possible row counts: 5, 5, 5, 5, 6 (sum 26) or 6, 5, 6, 5, 4?
    # A compact shape is better. 
    # Let's try 5 rows. 
    # Rows: 5, 5, 5, 5, 6? 
    # Hexagonal packing implies shifted rows.
    # Let's create a list of (row_index, col_index) or just coordinates.
    
    # Approximate radius for 26 circles in hex packing ~ 0.105
    # Grid spacing approx 2r ~ 0.21, vertical ~ 0.18.
    # Let's estimate positions.
    
    centers = np.zeros((n_circles, 2))
    idx = 0
    
    # Let's try a configuration with 5 rows.
    # Row 0: 6 circles
    # Row 1: 5 circles
    # Row 2: 6 circles
    # Row 3: 5 circles
    # Row 4: 4 circles -> Total 26.
    # Wait, 6+5+6+5+4 = 26.
    # Let's refine row lengths to be more compact.
    # Maybe 5, 5, 5, 5, 6?
    # Or 6, 6, 5, 5, 4?
    # Let's just place them in a hexagonal grid pattern and trim/adjust if needed,
    # but 26 is specific.
    
    # Let's try to fit them in a 6x5 area roughly.
    # Let's use a simple loop to generate hex grid points and pick 26 closest to center?
    # Or just predefined rows.
    
    # Let's try rows with counts: 6, 5, 6, 5, 4 (Total 26)
    # Row 0 (even): 6 circles. x starts at some offset.
    # Row 1 (odd): 5 circles. shifted.
    # Row 2 (even): 6 circles.
    # Row 3 (odd): 5 circles.
    # Row 4 (even): 4 circles.
    
    # Let's assume a radius r_est = 0.1 for placement.
    r_est = 0.1
    spacing_x = 2.0 * r_est
    spacing_y = np.sqrt(3) * r_est
    
    # We need to center this block in [0,1]x[0,1]
    # But optimization will move them.
    
    row_counts = [6, 5, 6, 5, 4]
    
    for i, count in enumerate(row_counts):
        # Vertical position
        y = r_est + i * spacing_y
        
        # Horizontal position
        # Even rows (0, 2, 4) start at x = r_est
        # Odd rows (1, 3) start at x = 2*r_est (shifted by r_est)
        if i % 2 == 0:
            start_x = r_est
        else:
            start_x = 2.0 * r_est
            
        for j in range(count):
            if idx < n_circles:
                x = start_x + j * spacing_x
                centers[idx] = [x, y]
                idx += 1
    
    # 2. Optimization
    # We want to maximize sum of radii.
    # We can optimize centers (x, y) and radii r.
    # However, variable radii is complex. Let's optimize for equal radii r first.
    # We maximize r such that circles fit.
    # This is equivalent to minimizing the violation of constraints.
    
    # Variables: [x1, y1, ..., x26, y26, r] -> 53 variables.
    # Actually, let's just optimize centers to minimize a "pressure" function
    # that tries to push circles apart. Then estimate r.
    # But better: optimize centers and r simultaneously using a penalty method.
    
    # Let's define a function to calculate penalty.
    # Penalty = sum of max(0, 2r - dist)^2 + boundary penalties.
    # We want to minimize penalty. But we want to maximize r.
    # This is tricky.
    
    # Alternative: Fixed r optimization.
    # Find max r by binary search?
    # For a fixed r, check if feasible.
    # Feasibility check is non-convex.
    
    # Let's use a simulated annealing or simple gradient descent on a potential function.
    # Potential U = -sum(r_i) + lambda * (overlap_penalty + boundary_penalty).
    # But if we assume equal radii, we maximize r.
    
    # Let's try to optimize centers to maximize the minimum distance between circles and boundaries.
    # Let d_min = min( min_ij(dist_ij - 2r), min_i(dist_to_wall - r) ).
    # We want to maximize d_min.
    # Actually, if we fix r, we want to check if d_min >= 0.
    # So we can maximize the clearance.
    
    # Let's assume equal radii r.
    # We can optimize centers to maximize the minimum clearance C.
    # C = min( (dist_ij)/2 - r, min_wall_dist - r ).
    # Wait, r is a variable.
    # Let's just optimize centers and r.
    
    # Using scipy minimize with method 'L-BFGS-B' or 'Nelder-Mead'
    # Variables: 26*2 centers + 1 radius = 53 vars.
    # Bounds: x,y in [0,1], r in [0, 0.5].
    
    # Objective: Minimize -r (maximize r)
    # Constraints:
    # x_i >= r, 1-x_i >= r
    # y_i >= r, 1-y_i >= r
    # dist_ij >= 2r
    
    # These are non-linear constraints.
    # We can use a penalty function approach inside the objective.
    
    def objective(vars):
        # vars: [x1, y1, ..., x26, y26, r]
        # Reshape
        c = vars[:52].reshape(26, 2)
        r = vars[52]
        
        if r < 0:
            return 1e6 # Penalty
        
        # Boundary penalty
        # We want x >= r, x <= 1-r => r <= x <= 1-r
        # Penalty if violated.
        penalty = 0.0
        
        # Wall constraints
        for i in range(26):
            x, y = c[i]
            # Left/Right
            if x - r < 0:
                penalty += 1000 * (r - x)**2
            if x + r > 1:
                penalty += 1000 * (x + r - 1)**2
            # Top/Bottom
            if y - r < 0:
                penalty += 1000 * (r - y)**2
            if y + r > 1:
                penalty += 1000 * (y + r - 1)**2
        
        # Overlap constraints
        # dist >= 2r
        for i in range(26):
            for j in range(i + 1, 26):
                dist = np.sqrt((c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2)
                min_dist = 2 * r
                if dist < min_dist:
                    penalty += 1000 * (min_dist - dist)**2
                    
        # We want to maximize r, so minimize -r + penalty
        # To balance, maybe scale r? r is around 0.1.
        return -r + penalty

    # Initial guess
    x0 = np.zeros(53)
    x0[:52] = centers.flatten()
    x0[52] = 0.1 # Initial radius guess
    
    # Bounds
    bounds = []
    for i in range(26):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    bounds.append((0.0, 0.5)) # r
    
    # Optimization
    # L-BFGS-B supports bounds.
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 10000})
    
    # Extract result
    opt_centers = res.x[:52].reshape(26, 2)
    opt_r = res.x[52]
    
    # The objective function tries to satisfy constraints.
    # If penalty is high, constraints are violated.
    # We can try multiple random restarts to avoid local minima.
    
    best_sum_radii = 26 * opt_r
    best_centers = opt_centers
    best_radii = np.full(26, opt_r)
    
    # Run a few more restarts
    for _ in range(10):
        # Random perturbation of centers
        random_centers = centers.copy() + np.random.normal(0, 0.02, centers.shape)
        # Clip to box
        random_centers = np.clip(random_centers, 0, 1)
        
        x0_temp = np.zeros(53)
        x0_temp[:52] = random_centers.flatten()
        x0_temp[52] = 0.1
        
        res_temp = minimize(objective, x0_temp, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000})
        
        if res_temp.fun < res.fun: # Lower objective is better (more negative or less penalty)
            res = res_temp
            opt_centers = res.x[:52].reshape(26, 2)
            opt_r = res.x[52]
            
    # Final check and adjustment
    # The optimization might result in slight violations if penalty isn't strong enough or local min.
    # We can try to shrink r slightly to ensure validity, but the function returns best found.
    # However, the validation function is strict.
    # We should ensure validity.
    
    # Let's compute the actual valid radii for the optimized centers.
    # For each circle, max radius is limited by walls and neighbors.
    # This is a system of constraints.
    # But since we assumed equal radii in optimization, we can just take the min feasible radius.
    
    # Calculate max feasible radius for this configuration
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    # r <= (dist_ij - r_j)/2 ? No, if all r equal, 2r <= dist_ij => r <= dist_ij/2.
    
    # So r_valid = min( min_i(min(x_i, 1-x_i, y_i, 1-y_i)), min_{i<j}(dist_ij/2) )
    
    dists = []
    for i in range(26):
        for j in range(i+1, 26):
            d = np.sqrt(np.sum((opt_centers[i] - opt_centers[j])**2))
            dists.append(d/2)
    
    wall_dists = []
    for i in range(26):
        x, y = opt_centers[i]
        wall_dists.append(min(x, 1-x, y, 1-y))
        
    r_valid = min(min(dists), min(wall_dists))
    
    # If r_valid is significantly smaller than opt_r, the optimizer failed to find a feasible point or penalty wasn't enough.
    # But with 1000 weight, it should be close.
    # We can output the valid packing.
    
    # To improve sum of radii, we can allow unequal radii.
    # With centers fixed, we can solve a small LP or just iterate.
    # But for now, equal radii is a strong baseline.
    # If r_valid is good, we are done.
    
    # Let's check if we can improve by slightly adjusting centers to maximize r_valid directly.
    # Actually, the objective was -r + penalty. If penalty ~ 0, -r is minimized => r maximized.
    # So opt_r should be close to r_valid.
    
    # Let's use r_valid to be safe.
    final_radii = np.full(26, r_valid)
    
    # Wait, if we use r_valid, sum is 26 * r_valid.
    # The optimizer returned opt_r. If constraints satisfied, opt_r == r_valid.
    
    return opt_centers, final_radii, 26 * r_valid