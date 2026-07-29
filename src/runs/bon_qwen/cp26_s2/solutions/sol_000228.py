# sol_000228 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b088ff81) state=fa0c3ab6 sum of radii=2.552821 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # 1. Initialize centers using a hexagonal pattern
    # We distribute 26 points in a hexagonal lattice within the square.
    # This is a dense packing heuristic.
    centers_init = []
    
    # Parameters for hexagonal grid
    # We try to fit roughly sqrt(N) x sqrt(N) grid with offset
    # Let's aim for a 6x5 or 5x6 structure roughly.
    # 26 points.
    
    # A simple way to generate a hexagonal grid:
    # Rows with alternating offsets.
    # Let's determine rows and cols.
    # Approx density: 26 points.
    # Let's try 6 rows. 
    # Col counts could be 5, 5, 5, 5, 5, 1? No, regular hex grid is better.
    # 6 rows of roughly 4-5 circles.
    # 4+5+4+5+4+4 = 26?
    
    # Let's just use a standard linspace and offset odd rows.
    # 5 columns, 6 rows = 30 points (too many).
    # 4 columns, 7 rows = 28 points.
    # 5 columns, 5 rows = 25 points. Add 1.
    
    # Let's use a 5x5 grid plus one extra point, shifted to fit hexagonally.
    # Or better: Generate points in a rectangle and scale to fit.
    
    # Hexagonal packing density optimization usually aligns with lattice vectors.
    # Let's create a grid of 6 columns and 5 rows, but remove 4 points?
    # Actually, just filling a rectangle with hex spacing is easiest.
    
    # Let's try 6 columns, 5 rows. Total 30 slots. We need 26.
    # We will fill the first 26 slots.
    
    # Grid parameters
    n_cols = 6
    n_rows = 5
    # Spacing. To fit in [0,1], we need to scale.
    # Horizontal spacing dx, vertical spacing dy = dx * sqrt(3)/2?
    # For hexagonal, distance between centers is 2r.
    # But we don't know r yet.
    # Let's just place points uniformly with hex offsets.
    
    # We want to maximize the minimum distance between points initially.
    # Let's place points in a grid that fits [0,1]x[0,1] tightly.
    
    # Let's use a simpler initialization:
    # Place points on a 5x5 grid (25 points) + 1 point in center?
    # No, hexagonal is better.
    
    # Let's generate coordinates for a hexagonal lattice
    # dx = 1.0 / (n_cols - 0.5)  # Approx spacing
    # dy = dx * np.sqrt(3) / 2.0
    
    # Actually, let's just use a random shuffle of a dense grid or a specific layout.
    # A robust layout:
    # 5 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 6 circles?
    # Total 26.
    
    # Let's define positions explicitly for a hexagonal-like packing.
    # We want to fit in [0,1]x[0,1].
    # Let's assume a target radius r_target approx 0.1.
    # Diameter 0.2.
    # 5 circles width 1.0. 5 circles height 1.0.
    # Hexagonal offset: rows shift by r.
    
    # Let's construct centers manually for a good start.
    # 5 rows, varying number of columns.
    # 6, 5, 6, 5, 4? Sum = 26.
    # Or 5, 6, 5, 6, 4?
    # Let's do 5 rows.
    # y coordinates: 0.1, 0.3, 0.5, 0.7, 0.9 (spacing 0.2)
    # Hexagonal offset x: 0.1, 0.3, ... or 0.2, 0.4...
    
    # Let's try:
    # Row 0 (y=0.1): 5 circles, x = 0.1, 0.3, 0.5, 0.7, 0.9
    # Row 1 (y=0.3): 5 circles, x = 0.2, 0.4, 0.6, 0.8, 1.0 (Wait, 1.0+r > 1)
    # Adjust to fit.
    
    # Let's use an algorithmic generation for a hexagonal grid that fits in [0,1]x[0,1]
    # with some margin, then the optimizer will expand.
    
    # Generate points on a hex lattice
    points = []
    # Lattice parameters
    # We want to pack 26 points.
    # Let's try a grid of width 1 and height 1.
    # dx = 1.0 / 5.0 # 0.2
    # dy = dx * np.sqrt(3) / 2.0 # ~0.1732
    
    # If we use dx=0.2, dy=0.1732.
    # Rows at y = 0.1, 0.273, 0.446, 0.620, 0.794, 0.967?
    # 6 rows.
    # Cols at x = 0.1, 0.3, 0.5, 0.7, 0.9 (5 cols).
    # 6*5 = 30 points. Too many.
    # We can just take the first 26.
    
    dx = 0.22
    dy = dx * np.sqrt(3) / 2.0
    
    y_curr = 0.12
    row_idx = 0
    while len(points) < n:
        # Determine x start for this row
        # Even rows start at 0.12, odd rows shifted by dx/2?
        # Standard hex: (i*dx, j*dy) and (i*dx + dx/2, j*dy + dy) ?
        # Actually, usually offset by half spacing.
        
        x_start = 0.12
        if row_idx % 2 == 1:
            x_start += dx / 2.0
            
        x_curr = x_start
        while x_curr <= 1.0 - 0.12 and len(points) < n:
            points.append([x_curr, y_curr])
            x_curr += dx
        y_curr += dy
        row_idx += 1
        
    centers_init = np.array(points[:n])
    
    # Initial radii: small value to ensure feasibility
    r_init = np.full(n, 0.02)
    
    # Combine into optimization variable vector
    # Order: x1, y1, r1, x2, y2, r2, ...
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = r_init[i]
        
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Constraints
    # 1. Boundary constraints for each circle
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => x + r - 1 <= 0
    # y - r >= 0  => r - y <= 0
    # y + r <= 1  => y + r - 1 <= 0
    
    # 2. Non-overlap
    # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
    # (ri + rj)^2 - (xi - xj)^2 - (yi - yj)^2 <= 0
    
    cons = []
    
    # Helper to access variables
    def get_x(idx): return x0[3*idx]
    def get_y(idx): return x0[3*idx+1]
    def get_r(idx): return x0[3*idx+2]
    
    # But constraints need to be functions of the full vector 'vars'
    
    def boundary_constraints(vars):
        c_vals = []
        for i in range(n):
            xi = vars[3*i]
            yi = vars[3*i+1]
            ri = vars[3*i+2]
            
            # r <= x
            c_vals.append(ri - xi)
            # r <= 1 - x  => x + r <= 1
            c_vals.append(xi + ri - 1.0)
            # r <= y
            c_vals.append(ri - yi)
            # r <= 1 - y => y + r <= 1
            c_vals.append(yi + ri - 1.0)
        return c_vals

    def overlap_constraints(vars):
        c_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                xi = vars[3*i]
                yi = vars[3*i+1]
                ri = vars[3*i+2]
                
                xj = vars[3*j]
                yj = vars[3*j+1]
                rj = vars[3*j+2]
                
                # Constraint: (ri + rj)^2 <= (xi-xj)^2 + (yi-yj)^2
                # Form: (ri + rj)^2 - dist^2 <= 0
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r_sq = (ri + rj)**2
                
                c_vals.append(sum_r_sq - dist_sq)
        return c_vals

    # To pass to scipy, we need a list of constraint dictionaries
    # {'type': 'ineq', 'fun': lambda v: -boundary_constraints(v)} ?
    # scipy.optimize.minimize expects g(x) >= 0 for 'ineq'.
    # Our constraints are <= 0. So we return -constraint.
    
    # However, passing a function that returns an array is supported in newer scipy?
    # Or we can pass a list of dicts.
    # List of dicts is safer.
    
    # But constructing 26*4 + 325 constraints explicitly as dicts is verbose.
    # We can use a single constraint function that returns a vector if we use a method that supports it?
    # SLSQP supports vectorized constraints? 
    # Actually, in scipy, if fun returns an array, it's treated as multiple constraints.
    # But the documentation says for 'ineq', it should return an array.
    
    # Let's define a wrapper
    def all_constraints(vars):
        # Boundary: r - x <= 0  -> -(r - x) >= 0  -> x - r >= 0
        # x + r - 1 <= 0 -> 1 - x - r >= 0
        # r - y <= 0 -> y - r >= 0
        # y + r - 1 <= 0 -> 1 - y - r >= 0
        
        # Overlap: (r+r)^2 - d^2 <= 0 -> d^2 - (r+r)^2 >= 0
        
        vals = []
        
        # Boundary
        for i in range(n):
            xi = vars[3*i]
            yi = vars[3*i+1]
            ri = vars[3*i+2]
            
            vals.append(xi - ri)       # x - r >= 0
            vals.append(1.0 - xi - ri) # 1 - x - r >= 0
            vals.append(yi - ri)       # y - r >= 0
            vals.append(1.0 - yi - ri) # 1 - y - r >= 0
            
        # Overlap
        for i in range(n):
            for j in range(i + 1, n):
                xi = vars[3*i]
                yi = vars[3*i+1]
                ri = vars[3*i+2]
                
                xj = vars[3*j]
                yj = vars[3*j+1]
                rj = vars[3*j+2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                
                # We want dist_sq >= sum_r^2
                # So dist_sq - sum_r^2 >= 0
                vals.append(dist_sq - sum_r**2)
                
        return np.array(vals)

    constraint_dict = {
        'type': 'ineq',
        'fun': all_constraints
    }
    
    # Objective: Maximize sum of radii => Minimize -sum(radii)
    def objective(vars):
        s = 0.0
        for i in range(n):
            s += vars[3*i + 2]
        return -s
    
    # Run optimization
    # Use SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraint_dict, 
                   options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})
    
    # Extract results
    if res.success or res.status == 0: # 0 is usually success in SLSQP? Actually check message
        # Sometimes SLSQP doesn't converge perfectly but gives a good result
        pass
    
    centers_opt = np.zeros((n, 2))
    radii_opt = np.zeros(n)
    
    for i in range(n):
        centers_opt[i, 0] = res.x[3*i]
        centers_opt[i, 1] = res.x[3*i+1]
        radii_opt[i] = res.x[3*i+2]
        
    sum_radii = np.sum(radii_opt)
    
    # Post-processing: Ensure radii are non-negative and centers valid
    # The optimizer should have handled bounds, but clip just in case
    radii_opt = np.maximum(radii_opt, 0.0)
    
    # Check if we can improve by just increasing radii slightly if feasible?
    # The optimizer maximizes sum, so it should be at the boundary.
    
    return centers_opt, radii_opt, float(sum_radii)
