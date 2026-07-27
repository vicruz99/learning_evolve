# sol_000034 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8e46300b) state=6d1ad14f sum of radii=2.516880 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal-like lattice
    # We try to fit 26 circles. 
    # A 5x5 grid fits 25. We need 1 more.
    # Hexagonal packing is denser.
    # Let's try 6 rows. 
    # 4, 5, 4, 5, 4, 4 -> 26 circles.
    # Or 5, 5, 5, 5, 4, 2 -> 26.
    # Let's create a general hexagonal grid and pick the first 26 points.
    
    # Parameters for initial grid
    # We start with a small radius to ensure no overlap, then expand.
    r_start = 0.08
    dx = 2 * r_start
    dy = math.sqrt(3) * r_start
    
    points = []
    row = 0
    while len(points) < n_circles:
        y = r_start + row * dy
        # Check if y is within bounds (roughly)
        if y + r_start > 1.0:
            break
            
        # Offset for hexagonal packing
        offset = (row % 2) * (dx / 2.0)
        x = r_start + offset
        
        while x + r_start <= 1.0:
            points.append([x, y])
            if len(points) == n_circles:
                break
            x += dx
        row += 1
        
    # If we didn't get enough points (unlikely with these params), pad with center
    while len(points) < n_circles:
        points.append([0.5, 0.5])
        
    # If we have more, trim
    points = points[:n_circles]
    
    # Convert to numpy array
    centers = np.array(points)
    
    # 2. Optimization
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # Total variables = 3 * n_circles
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = 0.05 # Initial radius guess
    
    def objective(vars):
        # We want to maximize sum of radii -> minimize -sum(radii)
        r = vars[2::3]
        return -np.sum(r)
    
    def constraints(vars):
        cons = []
        
        # Extract coordinates and radii
        cx = vars[0::3]
        cy = vars[1::3]
        r = vars[2::3]
        
        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        # x - r >= 0  => x - r >= 0
        # 1 - x - r >= 0 => 1 - x - r >= 0
        # Same for y
        
        for i in range(n_circles):
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]}) # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]}) # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]}) # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]}) # 1 - y - r >= 0
            
            # Radius non-negativity (strictly positive usually better, but >= 0 is required)
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})

        # Non-overlap constraints: dist >= r_i + r_j
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                # To avoid lambda closure issues, define helper or use index directly
                # But defining inside loop with lambda is tricky with index.
                # We can just append the constraint dict with a nested function or use a list of functions.
                # However, scipy accepts a list of dicts.
                # Let's create a specific function for this pair.
                
                def overlap_func(v, i=i, j=j):
                    dx = v[3*i] - v[3*j]
                    dy = v[3*i+1] - v[3*j+1]
                    ri = v[3*i+2]
                    rj = v[3*j+2]
                    return (dx**2 + dy**2) - (ri + rj)**2
                
                cons.append({'type': 'ineq', 'fun': overlap_func})
        
        return cons

    # Solve
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    try:
        result = scipy.optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda v: -1.0}, # Dummy to pass args if needed, but we use list
            # Actually, scipy.optimize.minimize with SLSQP accepts constraints as a list of dicts directly?
            # No, it expects a NonlinearConstraint or dict. 
            # For SLSQP, constraints can be a list of dictionaries.
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        # Wait, the `constraints` argument in minimize expects a constraint spec.
        # If we generate constraints dynamically, we can't pass them easily as a list to minimize directly in older scipy versions?
        # Actually, standard usage: constraints=[{'type': 'ineq', 'fun': f1}, ...] works in SLSQP.
        # But the functions inside must be serializable or accessible.
        # Let's rewrite to use a single vectorized constraint function if possible, or just use the list.
        
        # Re-evaluating constraints approach for SLSQP:
        # It is cleaner to define a function that returns the constraint vector.
        
        def constraint_func(vars):
            cx = vars[0::3]
            cy = vars[1::3]
            r = vars[2::3]
            
            vals = []
            # Boundary
            vals.extend(cx - r)
            vals.extend(1.0 - cx - r)
            vals.extend(cy - r)
            vals.extend(1.0 - cy - r)
            vals.extend(r) # r >= 0
            
            # Overlaps
            # Vectorized calculation
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    ri = r[i]
                    rj = r[j]
                    d2 = dx*dx + dy*dy
                    sum_r = ri + rj
                    vals.append(d2 - sum_r*sum_r)
            
            return np.array(vals)
        
        # We need to pass this to minimize.
        # However, SLSQP might struggle with hundreds of constraints if not careful, but 26 circles -> ~325 overlap constraints + 104 boundary. Total ~430.
        # This is manageable.
        
        # Using a wrapper for constraints to pass to minimize
        # Actually, for SLSQP, we can pass constraints as a list of dicts where fun is the scalar function.
        # But creating 400 dicts is verbose.
        # Better: Use a single function that returns a vector of constraints >= 0.
        # scipy.optimize.minimize supports 'NonlinearConstraint' but that's for 'eq' and 'ineq' vectorized?
        # Actually, standard 'ineq' expects a scalar.
        # Let's stick to the list of dicts approach but generate it cleanly.
        
        cons_list = []
        
        # Boundary constraints
        for i in range(n_circles):
            # x >= r
            cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
            # 1 - x >= r
            cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
            # y >= r
            cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
            # 1 - y >= r
            cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
            # r >= 0
            cons_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})

        # Overlap constraints
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                cons_list.append({
                    'type': 'ineq', 
                    'fun': lambda v, i=i, j=j: (v[3*i]-v[3*j])**2 + (v[3*i+1]-v[3*j+1])**2 - (v[3*i+2]+v[3*j+2])**2
                })

        result = scipy.optimize.minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons_list,
            options={'maxiter': 2000, 'ftol': 1e-12}
        )
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        # Fallback to initial
        pass

    # 3. Extract results
    best_vars = result.x
    final_centers = np.zeros((n_circles, 2))
    final_radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        final_centers[i, 0] = best_vars[3*i]
        final_centers[i, 1] = best_vars[3*i+1]
        final_radii[i] = best_vars[3*i+2]
        
    sum_radii = np.sum(final_radii)
    
    # 4. Validation and Adjustment
    # The optimizer might produce a solution with tiny violations due to tolerance.
    # We should ensure strict validity.
    
    # Check for violations and shrink radii slightly if needed
    # Or just rely on the optimizer. 
    # Let's do a safety shrink.
    
    # Calculate min distance to boundary and neighbors for each circle
    # To be safe, we can scale down radii by a very small epsilon if any violation is detected.
    
    # Quick check
    valid = True
    for i in range(n_circles):
        x, y = final_centers[i]
        r = final_radii[i]
        if x < r or x > 1-r or y < r or y > 1-r:
            valid = False
            break
        for j in range(i + 1, n_circles):
            x2, y2 = final_centers[j]
            r2 = final_radii[j]
            dist = np.sqrt((x-x2)**2 + (y-y2)**2)
            if dist < r + r2:
                valid = False
                break
        if not valid: break
    
    if not valid:
        # If invalid, apply a uniform shrinkage factor to radii until valid
        # This is a simple recovery
        shrink_factor = 1.0
        for _ in range(100):
            if valid: break
            valid = True
            final_radii_scaled = final_radii * shrink_factor
            # Check
            for i in range(n_circles):
                x, y = final_centers[i]
                r = final_radii_scaled[i]
                if x < r or x > 1-r or y < r or y > 1-r:
                    valid = False
                    break
                for j in range(i + 1, n_circles):
                    x2, y2 = final_centers[j]
                    r2 = final_radii_scaled[j]
                    dist = np.sqrt((x-x2)**2 + (y-y2)**2)
                    if dist < r + r2:
                        valid = False
                        break
                if not valid: break
            if not valid:
                shrink_factor *= 0.99
        final_radii = final_radii * shrink_factor
        sum_radii = np.sum(final_radii)

    # Ensure no NaNs
    if np.isnan(final_centers).any() or np.isnan(final_radii).any():
        # Fallback to a safe grid
        print("Fallback to grid packing")
        k = 5
        step = 1.0 / (k + 1)
        # Grid of 5x5 = 25. Need 26.
        # Let's place 25 in grid and 1 in center?
        # Better: Just use a small radius grid for 26
        # 6x5 grid? 30 spots.
        step_x = 1.0 / 7.0
        step_y = 1.0 / 5.0
        idx = 0
        final_centers = np.zeros((26, 2))
        final_radii = np.zeros(26)
        r_safe = 0.05
        y = step_y
        while idx < 26:
            x = step_x
            while x <= 1.0 - step_x and idx < 26:
                final_centers[idx] = [x, y]
                final_radii[idx] = r_safe
                x += step_x
                idx += 1
            y += step_y
        sum_radii = np.sum(final_radii)

    return final_centers, final_radii, float(sum_radii)
