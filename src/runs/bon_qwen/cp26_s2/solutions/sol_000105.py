# sol_000105 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 93a6f440) state=9148aca9 sum of radii=2.599728 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def generate_hexagonal_packing(n):
    """
    Generates an initial configuration of n centers in a hexagonal pattern.
    """
    # Try to fit n circles in a hexagonal grid pattern
    # We will iterate row by row
    centers = []
    # Approximate radius for initial placement (small enough to fit many)
    # We will scale later or just place them and let optimizer fix.
    # Let's place them with spacing 0.15 to be safe, then optimizer scales.
    spacing = 0.15
    x_spacing = spacing
    y_spacing = spacing * math.sqrt(3) / 2
    
    # We need to fit 26 circles. 
    # Let's try to fill the square [0,1]x[0,1] with hex points.
    # Row 0: y = spacing, x = spacing, 3*spacing, ...
    # Row 1: y = spacing + y_spacing, x = 2*spacing, ...
    
    # Actually, let's just generate a grid of potential points and pick the best n?
    # Or construct a specific block.
    # For 26, maybe 5 rows. 6, 5, 6, 5, 4?
    # Let's try a generic hex grid fill.
    
    points = []
    y = spacing
    row_idx = 0
    while y <= 1 - spacing:
        offset = spacing if row_idx % 2 == 1 else spacing
        # Wait, standard hex offset is half spacing?
        # If centers are at x, x+2r, shift is r.
        # Here spacing is distance between centers in row = 2r_approx.
        # Shift is r_approx = spacing/2.
        
        current_x = spacing
        if row_idx % 2 == 1:
            current_x = spacing * 1.5 # Shift by half spacing? 
            # If row spacing is 2r, shift is r.
            # Let's assume spacing = 2r_approx.
            # Row 0: r, 3r, 5r... -> x = r, r+2r...
            # Row 1: 2r, 4r... -> x = 2r, 2r+2r...
            # Shift is r = spacing/2.
            # Start of Row 0: spacing.
            # Start of Row 1: spacing + spacing/2 = 1.5 * spacing.
            pass 
        
        # Correct logic:
        # Base spacing dx = 0.2 (arbitrary small)
        # dy = dx * sqrt(3) / 2
        # Row 0: x = dx, 3dx, 5dx...
        # Row 1: x = 2dx, 4dx... (shifted by dx)
        
        dx = 0.18 # Initial guess for diameter
        dy = dx * math.sqrt(3) / 2
        
        # Reset generation with fixed dx
        pass

    # Let's use a fixed dx for initialization
    dx = 0.18
    dy = dx * math.sqrt(3) / 2
    
    y = dx
    row = 0
    while y <= 1 - dx:
        x = dx
        if row % 2 == 1:
            x = dx * 2 # Shift by dx (which is diameter)
            # Wait, if row 0 starts at dx, row 1 should start at 2*dx?
            # Row 0 centers: dx, 3dx, 5dx...
            # Row 1 centers: 2dx, 4dx...
            # Yes, shift is dx.
        
        while x <= 1 - dx:
            centers.append([x, y])
            x += dx * 2
        y += dy
        row += 1
        
    # If we have more than n, trim. If less, scale down dx and retry?
    # For 26, with dx=0.18, width ~ 1.
    # 1/0.18 ~ 5.5 columns.
    # Rows: 1/ (0.18*0.866) ~ 1/0.156 ~ 6.4 rows.
    # 5.5 * 6.4 ~ 35 points.
    # We need 26.
    # Just take the first 26? Or center them?
    # Taking first 26 from top-left might be biased.
    # Let's just take the first 26 generated.
    
    if len(centers) > n:
        centers = centers[:n]
    
    # If we have fewer, it means dx was too big.
    # But dx=0.18 should yield enough.
    
    return np.array(centers)

def optimize_equal_radii(centers_init, n):
    """
    Optimizes positions to maximize the radius of equal circles.
    """
    # Variables: x_1, y_1, ..., x_n, y_n, r
    # Total 2*n + 1 variables.
    # We stack them: [x1, y1, ..., xn, yn, r]
    
    # Initial r estimate
    # Based on centers, min dist to boundary and min pair dist / 2
    min_r = 1.0
    for i in range(n):
        x, y = centers_init[i]
        min_r = min(min_r, x, 1-x, y, 1-y)
    
    # Check pair distances (just for safety, though init should be valid)
    # We can just start with a small r and let optimizer increase.
    initial_r = 0.05 
    
    x0 = np.hstack([centers_init.flatten(), initial_r])
    
    def objective(vars):
        # Maximize sum of radii = n * r
        # vars[-1] is r
        return -n * vars[-1]

    def constraint_boundary(vars):
        # vars[2*i] = x, vars[2*i+1] = y
        # vars[-1] = r
        r = vars[-1]
        constrs = []
        for i in range(n):
            x = vars[2*i]
            y = vars[2*i+1]
            # x >= r, x <= 1-r, y >= r, y <= 1-r
            constrs.append(x - r)
            constrs.append(1 - r - x)
            constrs.append(y - r)
            constrs.append(1 - r - y)
        return np.array(constrs)

    def constraint_separation(vars):
        r = vars[-1]
        constrs = []
        for i in range(n):
            xi = vars[2*i]
            yi = vars[2*i+1]
            for j in range(i + 1, n):
                xj = vars[2*j]
                yj = vars[2*j+1]
                # dist >= 2r  => dist^2 >= 4r^2
                # (xi-xj)^2 + (yi-yj)^2 - 4r^2 >= 0
                d2 = (xi - xj)**2 + (yi - yj)**2
                constrs.append(d2 - 4 * r**2)
        return np.array(constrs)

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
    bounds.append((1e-6, 0.5)) # r

    # Constraints setup for SLSQP
    cons = []
    cons.append({'type': 'ineq', 'fun': constraint_boundary})
    cons.append({'type': 'ineq', 'fun': constraint_separation})

    # Run optimization
    # Use multiple restarts if necessary, but one run with good init should work.
    res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})
    
    if res.success or res.fun > -100: # Check if valid
        final_r = res.x[-1]
        final_centers = res.x[:-1].reshape((n, 2))
        return final_centers, np.full(n, final_r), res.fun * -1
    else:
        # Fallback to initial
        return centers_init, np.full(n, initial_r), initial_r * n

def optimize_variable_radii(centers_init, n):
    """
    Tries to optimize with variable radii.
    """
    # Variables: x1, y1, r1, ..., xn, yn, rn
    # 3*n variables.
    
    # Initial radii: estimate from equal radius optimization or small value
    # Let's start with small radii to be safe
    initial_r = 0.05
    
    x0 = []
    for i in range(n):
        x0.append(centers_init[i, 0])
        x0.append(centers_init[i, 1])
        x0.append(initial_r)
    x0 = np.array(x0)
    
    def objective(vars):
        # Sum of radii is at indices 2, 5, 8, ...
        r_sum = sum(vars[2 + 3*i] for i in range(n))
        return -r_sum

    def constraint_boundary(vars):
        constrs = []
        for i in range(n):
            x = vars[3*i]
            y = vars[3*i+1]
            r = vars[3*i+2]
            constrs.append(x - r)
            constrs.append(1 - r - x)
            constrs.append(y - r)
            constrs.append(1 - r - y)
        return np.array(constrs)

    def constraint_separation(vars):
        constrs = []
        for i in range(n):
            xi = vars[3*i]
            yi = vars[3*i+1]
            ri = vars[3*i+2]
            for j in range(i + 1, n):
                xj = vars[3*j]
                yj = vars[3*j+1]
                rj = vars[3*j+2]
                # dist >= ri + rj
                # dist^2 >= (ri+rj)^2
                d2 = (xi - xj)**2 + (yi - yj)**2
                constrs.append(d2 - (ri + rj)**2)
        return np.array(constrs)

    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((1e-6, 0.5)) # r

    cons = []
    cons.append({'type': 'ineq', 'fun': constraint_boundary})
    cons.append({'type': 'ineq', 'fun': constraint_separation})

    try:
        res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'ftol': 1e-9, 'maxiter': 2000, 'disp': False})
        
        if res.success:
            centers = np.zeros((n, 2))
            radii = np.zeros(n)
            for i in range(n):
                centers[i] = [res.x[3*i], res.x[3*i+1]]
                radii[i] = res.x[3*i+2]
            return centers, radii, -res.fun
    except Exception:
        pass
        
    # Fallback
    return centers_init, np.full(n, initial_r), initial_r * n

def run_packing():
    n = 26
    
    # 1. Generate initial hexagonal packing
    # The function generate_hexagonal_packing creates a grid.
    # We need to make sure it produces a valid starting point for the optimizer.
    # The optimizer handles constraints, so even if initial points overlap, 
    # as long as r is small enough, it's fine.
    
    # Let's refine the initialization logic inside here to be robust.
    centers_init = np.zeros((n, 2))
    
    # Simple initialization: place in a grid, then optimize.
    # Hexagonal grid initialization:
    dx = 0.15
    dy = dx * math.sqrt(3) / 2
    row = 0
    count = 0
    y = dx
    while count < n and y <= 1 - dx:
        x = dx
        if row % 2 == 1:
            x = dx * 2 # Shift
        while count < n and x <= 1 - dx:
            centers_init[count, 0] = x
            centers_init[count, 1] = y
            count += 1
            x += dx * 2
        y += dy
        row += 1
    
    # If we didn't fill all, fill remaining randomly or just leave them?
    # The while loop should fill 26 easily with dx=0.15.
    # dx=0.15 allows ~6 cols, ~7 rows. 42 points. We take first 26.
    
    # 2. Optimize equal radii
    # This is the most robust step to get a high sum.
    centers_eq, radii_eq, sum_eq = optimize_equal_radii(centers_init, n)
    
    # Validate the result of equal radii optimization
    # Sometimes SLSQP might violate constraints slightly if not converged perfectly,
    # but with 'ineq' constraints it should be feasible.
    # Let's do a quick check.
    if validate_packing(centers_eq, radii_eq):
        current_sum = sum_eq
        best_centers = centers_eq
        best_radii = radii_eq
    else:
        # If validation fails, we might need to reduce radii slightly
        # to satisfy numerical tolerance.
        min_violation = 0.0
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(centers_eq[i] - centers_eq[j])
                req = radii_eq[i] + radii_eq[j]
                if dist < req:
                    min_violation = max(min_violation, req - dist)
        
        # Reduce radii by half violation + epsilon
        correction = min_violation / 2 + 1e-4
        best_radii = radii_eq - correction
        best_centers = centers_eq
        current_sum = np.sum(best_radii)
        
    # 3. Try variable radii optimization using the equal radii solution as seed
    # This might yield a better sum.
    centers_var, radii_var, sum_var = optimize_variable_radii(best_centers, n)
    
    if validate_packing(centers_var, radii_var):
        if sum_var > current_sum:
            best_centers = centers_var
            best_radii = radii_var
            current_sum = sum_var

    # Final validation and safety correction
    # Ensure strict validity
    is_valid = validate_packing(best_centers, best_radii)
    if not is_valid:
        # Emergency shrink
        # Find min distance between centers
        min_dist = 1.0
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                if d < min_dist:
                    min_dist = d
        
        # Max possible radius is min_dist / 2 (if touching)
        # Also boundary constraints.
        max_r = min_dist / 2
        
        # Check boundaries
        for i in range(n):
            x, y = best_centers[i]
            max_r = min(max_r, x, 1-x, y, 1-y)
        
        # If max_r is very small, something went wrong.
        # But with valid centers, max_r > 0.
        
        # If current radii are larger than max_r, shrink them.
        if np.any(best_radii > max_r - 1e-9):
            # Uniformly shrink? Or individual?
            # Just set all to max_r to be safe? No, that kills sum.
            # Just ensure validity.
            # Let's scale down radii until valid.
            scale = 1.0
            for i in range(n):
                # Boundary
                scale = min(scale, best_centers[i, 0] / best_radii[i] if best_radii[i] > 0 else 1)
                scale = min(scale, (1 - best_centers[i, 0]) / best_radii[i] if best_radii[i] > 0 else 1)
                scale = min(scale, best_centers[i, 1] / best_radii[i] if best_radii[i] > 0 else 1)
                scale = min(scale, (1 - best_centers[i, 1]) / best_radii[i] if best_radii[i] > 0 else 1)
            
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(best_centers[i] - best_centers[j])
                    if dist > 0:
                        scale = min(scale, dist / (best_radii[i] + best_radii[j]) if (best_radii[i] + best_radii[j]) > 0 else 1)
            
            # Apply scale (with safety factor)
            scale *= 0.999
            best_radii *= scale
            current_sum = np.sum(best_radii)
            # Re-validate?
            # Ideally it should be valid now.

    return best_centers, best_radii, np.sum(best_radii)
