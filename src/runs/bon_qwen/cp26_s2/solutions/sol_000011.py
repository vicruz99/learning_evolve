# sol_000011 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b5cb09ab) state=f0122509 sum of radii=2.080000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    # Number of circles
    N = 26
    
    # Initial guess: Place circles in a grid pattern
    # We want to fit 26 circles. Let's try a 6x5 grid subset or similar.
    # Or a 5x5 grid with one extra.
    # Let's create a 5 row, 6 col grid (30 spots) and pick 26?
    # Or just arrange them nicely.
    
    # Let's try a hexagonal-ish initialization.
    # 5 rows.
    # Row counts: 6, 5, 6, 5, 4 -> Sum 26.
    # Or 5, 6, 5, 6, 4 -> 26.
    # Let's try 5, 6, 5, 6, 4.
    
    # Row y-coordinates
    # If we assume r approx 0.1, height 1.
    # 5 rows -> spacing 0.2. y = 0.1, 0.3, 0.5, 0.7, 0.9.
    
    rows = 5
    row_counts = [5, 6, 5, 6, 4] # Sum = 26
    
    centers_init = []
    radii_init = []
    
    # Approximate radius for initialization. 
    # If we have 6 circles in a row, width 1 implies 2*r*6 <= 1 => r <= 1/12 ~ 0.083.
    # Let's start with r = 0.08 to be safe.
    r_start = 0.08
    
    # Y positions for 5 rows
    # We want to span [r, 1-r].
    # Let's just use uniform spacing.
    y_positions = np.linspace(0.1, 0.9, rows) # 0.1, 0.3, 0.5, 0.7, 0.9
    
    idx = 0
    for r_idx, count in enumerate(row_counts):
        y = y_positions[r_idx]
        # X positions
        # If row has 'count' circles, spread them in [0.1, 0.9] roughly?
        # Or [r, 1-r]. With r=0.08, [0.08, 0.92].
        # Spacing = (1 - 2*r_start) / (count - 1) if count > 1
        if count == 1:
            x_pos = 0.5
        else:
            # Center the row
            width_available = 1.0 - 2 * r_start
            spacing = width_available / (count - 1)
            x_start = r_start # + (1 - 2*r_start - (count-1)*spacing)/2
            # Actually, if we just spread evenly:
            # x = np.linspace(r_start, 1-r_start, count)
            # But for hex packing, odd/even rows should be offset.
            # Let's do simple grid first.
            x_pos = np.linspace(r_start, 1 - r_start, count)
        
        for x in x_pos:
            centers_init.append([x, y])
            radii_init.append(r_start)
            idx += 1
            
    centers_init = np.array(centers_init)
    radii_init = np.array(radii_init)
    
    # Combine into a single vector for optimization
    # Order: x1, y1, r1, x2, y2, r2, ...
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    # Constraints
    # 1. Boundary constraints: r <= x <= 1-r  => x - r >= 0, 1 - x - r >= 0
    #    Similarly for y.
    # 2. Non-overlap: dist(i, j) >= r_i + r_j
    #    (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    #    => (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    
    cons = []
    
    # Boundary constraints
    for i in range(N):
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        
    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            # dist^2 - (r_i + r_j)^2 >= 0
            def overlap(v, i=i, j=j):
                xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            cons.append({'type': 'ineq', 'fun': overlap})
            
    # Objective: Maximize sum of radii => Minimize -sum(r)
    def objective(v):
        total_r = 0.0
        for i in range(N):
            total_r += v[3*i+2]
        return -total_r
        
    # Run optimization
    # SLSQP is suitable for bound and non-linear constraints
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})
    
    if res.success:
        optimal_x = res.x
    else:
        # If failed, use initial guess or try to recover?
        # For safety, just use what we got or initial.
        optimal_x = x0 # Fallback
        
    # Extract results
    final_centers = np.zeros((N, 2))
    final_radii = np.zeros(N)
    
    for i in range(N):
        final_centers[i, 0] = optimal_x[3*i]
        final_centers[i, 1] = optimal_x[3*i+1]
        final_radii[i] = optimal_x[3*i+2]
        
    sum_radii = np.sum(final_radii)
    
    # Validation check (internal)
    # The validate_packing function in the prompt checks this.
    # We trust the optimizer mostly.
    
    return final_centers, final_radii, sum_radii
