import numpy as np
from scipy.optimize import minimize

def generate_hexagonal_grid(n):
    """
    Generates an initial hexagonal grid configuration for n circles.
    """
    centers = []
    r_guess = 0.08  # Initial guess, will be adjusted
    y = r_guess
    row = 0
    count = 0
    
    # Approximate number of rows
    # Area ~ n * pi * r^2 <= 1. 
    # Hex packing density pi/sqrt(12) ~ 0.9069.
    # n * pi * r^2 ~ 0.9 * 1 => r ~ sqrt(0.9 / (n*pi)) ~ 0.099
    # But we leave space for optimization.
    
    # Try to fit rows
    while count < n:
        # x starts at r_guess, increment by 2*r_guess
        # For odd rows, shift by r_guess
        x_start = r_guess if row % 2 == 0 else 2 * r_guess
        
        x = x_start
        while x <= 1 - r_guess:
            if count < n:
                centers.append([x, y])
                count += 1
            x += 2 * r_guess
        
        # Increment y by sqrt(3)*r
        y += np.sqrt(3) * r_guess
        row += 1
        
        # If y goes out of bounds, shrink r and restart grid generation logic 
        # or just append remaining? 
        # A simple grid generator might not perfectly fit n in a square with fixed r.
        # Let's do a simpler dense packing logic if this loop fails.
        if y > 1 + r_guess and count < n:
             # Fallback: just fill a grid if hex row logic runs out
             # But let's rely on the optimizer to move them.
             pass

    if count < n:
        # Fallback to random valid placement if grid wasn't enough
        # Fill remaining with random points, small radius
        while count < n:
            cx = np.random.uniform(0.1, 0.9)
            cy = np.random.uniform(0.1, 0.9)
            # Simple check to not overlap too much with existing
            dists = np.sqrt(np.sum((centers - np.array([cx, cy]))**2, axis=1))
            if np.min(dists) > 0.2:
                centers.append([cx, cy])
                count += 1
            else:
                # Force place with small radius
                centers.append([cx, cy])
                count += 1
                
    return np.array(centers), r_guess

def get_constraints(n):
    """
    Defines the constraints for the optimization problem.
    1. Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    2. Overlap constraints: dist^2 >= (r_i + r_j)^2
    """
    constraints = []
    
    # Boundary constraints
    # x_i - r_i >= 0
    # r_i - x_i <= 0  =>  x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    # We will pass these as non-linear constraints in the main function or bounds?
    # Bounds are easier for simple box constraints, but r is coupled with x.
    # x >= r is not a simple box bound on x because r varies.
    # So we use nonlinear constraints.
    
    # Helper to map index in flat vector to variable
    # Vector structure: [x1, y1, r1, x2, y2, r2, ...]
    # x_i at 3*(i-1), y_i at 3*(i-1)+1, r_i at 3*(i-1)+2 (0-indexed logic below)
    
    # Actually, let's use indices 0..n-1 for circles.
    # Variables: x[i], y[i], r[i]
    # Flat array index: 3*i, 3*i+1, 3*i+2
    
    # Boundary: x_i - r_i >= 0
    for i in range(n):
        idx_x = 3 * i
        idx_r = 3 * i + 2
        def boundary_x(i=i, idx_x=idx_x, idx_r=idx_r):
            return lambda v: v[idx_x] - v[idx_r]
        constraints.append({'type': 'ineq', 'fun': boundary_x})
        
        # 1 - x_i - r_i >= 0
        def boundary_x_1(i=i, idx_x=idx_x, idx_r=idx_r):
            return lambda v: 1.0 - v[idx_x] - v[idx_r]
        constraints.append({'type': 'ineq', 'fun': boundary_x_1})

        idx_y = 3 * i + 1
        def boundary_y(i=i, idx_y=idx_y, idx_r=idx_r):
            return lambda v: v[idx_y] - v[idx_r]
        constraints.append({'type': 'ineq', 'fun': boundary_y})

        def boundary_y_1(i=i, idx_y=idx_y, idx_r=idx_r):
            return lambda v: 1.0 - v[idx_y] - v[idx_r]
        constraints.append({'type': 'ineq', 'fun': boundary_y_1})

    # Overlap constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi, idx_yi, idx_ri = 3*i, 3*i+1, 3*i+2
            idx_xj, idx_yj, idx_rj = 3*j, 3*j+1, 3*j+2
            
            def overlap(i=i, j=j, 
                        idx_xi=idx_xi, idx_yi=idx_yi, idx_ri=idx_ri,
                        idx_xj=idx_xj, idx_yj=idx_yj, idx_rj=idx_rj):
                def func(v):
                    dx = v[idx_xi] - v[idx_xj]
                    dy = v[idx_yi] - v[idx_yj]
                    r_sum = v[idx_ri] + v[idx_rj]
                    return dx*dx + dy*dy - r_sum*r_sum
                return func
            constraints.append({'type': 'ineq', 'fun': overlap()})
            
    return constraints

def objective(v, n):
    # Maximize sum of radii -> Minimize -sum(r)
    radii = v[2::3]
    return -np.sum(radii)

def run_packing():
    n = 26
    
    # Initial guess generation
    # We try a few seeds to avoid local minima
    best_v = None
    best_val = -np.inf
    
    # Generate initial hex grid
    centers, r_init = generate_hexagonal_grid(n)
    
    # Create initial vector
    # [x1, y1, r1, x2, y2, r2, ...]
    v0 = np.zeros(3 * n)
    for i in range(n):
        v0[3*i] = centers[i][0]
        v0[3*i+1] = centers[i][1]
        v0[3*i+2] = r_init
    
    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (3 * n)
    for i in range(n):
        bounds[3*i+2] = (0, 0.5) # r bound
    
    # Constraints
    constraints = get_constraints(n)
    
    # Optimization options
    options = {
        'maxiter': 1000,
        'ftol': 1e-9,
        'disp': False
    }
    
    # Try multiple restarts with slight perturbations to escape local optima
    results = []
    for restart in range(5):
        # Perturb initial guess slightly
        current_v0 = v0.copy()
        # Add small noise to centers and radii
        noise = np.random.normal(0, 0.005, size=current_v0.shape)
        # Ensure radii stay positive and valid
        current_v0[2::3] = np.maximum(0.01, current_v0[2::3] + noise[2::3])
        current_v0[0::3] = np.clip(current_v0[0::3] + noise[0::3], 0.01, 0.99)
        current_v0[1::3] = np.clip(current_v0[1::3] + noise[1::3], 0.01, 0.99)
        
        # Try to optimize
        try:
            res = minimize(
                objective, 
                current_v0, 
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options=options
            )
            if res.success:
                results.append(res)
        except Exception:
            continue
            
    # Also try a purely random initialization
    for restart in range(5):
        v_rand = np.random.uniform(0.1, 0.9, size=(n, 2)).flatten()
        r_rand = 0.02 * np.ones(n) # Small radii to start valid
        # Interleave x, y, r
        v0_rand = np.zeros(3*n)
        v0_rand[0::3] = v_rand[0::2] # x
        v0_rand[1::3] = v_rand[1::2] # y
        v0_rand[2::3] = r_rand       # r
        
        try:
            res = minimize(
                objective, 
                v0_rand, 
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options=options
            )
            if res.success:
                results.append(res)
        except Exception:
            continue

    # Select best result
    best_result = None
    best_sum = -np.inf
    
    for res in results:
        sum_r = -res.fun # Because we minimized negative sum
        if sum_r > best_sum:
            # Verify validity manually to be safe
            v = res.x
            centers_out = np.zeros((n, 2))
            radii_out = np.zeros(n)
            for i in range(n):
                centers_out[i] = [v[3*i], v[3*i+1]]
                radii_out[i] = v[3*i+2]
            
            # Quick validation check (subset of validate_packing)
            valid = True
            for i in range(n):
                x, y = centers_out[i]
                r = radii_out[i]
                if x < r or x > 1-r or y < r or y > 1-r:
                    valid = False; break
                for j in range(i+1, n):
                    dist = np.sqrt((x-centers_out[j][0])**2 + (y-centers_out[j][1])**2)
                    if dist < radii_out[j] + r - 1e-9:
                        valid = False; break
                if not valid: break
            
            if valid:
                best_sum = sum_r
                best_result = (centers_out, radii_out, sum_r)
    
    if best_result is None:
        # Fallback to the last attempted result if all failed validation (shouldn't happen)
        # Or just return the best unverified sum
        print("Warning: No fully valid packing found in attempts. Returning best unverified.")
        # Pick best from results based on objective value
        best_res = min(results, key=lambda r: r.fun)
        v = best_res.x
        centers_out = np.zeros((n, 2))
        radii_out = np.zeros(n)
        for i in range(n):
            centers_out[i] = [v[3*i], v[3*i+1]]
            radii_out[i] = v[3*i+2]
        return centers_out, radii_out, -best_res.fun

    return best_result