# sol_000056 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 50e7db78) state=daa9db77 sum of radii=2.539937 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def objective(params, n):
    """
    Objective function to minimize: negative sum of radii.
    params: array of shape (n * 3) -> [x1, y1, r1, x2, y2, r2, ...]
    """
    radii = params[2::3]
    return -np.sum(radii)

def boundary_constraints(params, n):
    """
    Constraints for circles to be inside the unit square.
    r <= x <= 1-r  => x - r >= 0, x + r <= 1
    r <= y <= 1-r  => y - r >= 0, y + r <= 1
    """
    constraints = []
    for i in range(n):
        idx = 3 * i
        x = params[idx]
        y = params[idx + 1]
        r = params[idx + 2]
        
        # x - r >= 0
        constraints.append(x - r)
        # 1 - x - r >= 0
        constraints.append(1 - x - r)
        # y - r >= 0
        constraints.append(y - r)
        # 1 - y - r >= 0
        constraints.append(1 - y - r)
    return np.array(constraints)

def overlap_constraints(params, n):
    """
    Constraints for non-overlapping circles.
    dist(i, j)^2 - (ri + rj)^2 >= 0
    """
    constraints = []
    for i in range(n):
        idx_i = 3 * i
        xi = params[idx_i]
        yi = params[idx_i + 1]
        ri = params[idx_i + 2]
        
        for j in range(i + 1, n):
            idx_j = 3 * j
            xj = params[idx_j]
            yj = params[idx_j + 1]
            rj = params[idx_j + 2]
            
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            sum_r = ri + rj
            
            # Constraint: dist_sq - sum_r^2 >= 0
            # We use a small buffer or just the raw value
            # Note: SLSQP expects constraints >= 0
            constraints.append(dist_sq - sum_r**2)
            
    return np.array(constraints)

def radius_constraints(params, n):
    """
    Constraints for radii to be non-negative.
    r >= 0
    """
    return params[2::3]

def generate_hexagonal_init(n):
    """
    Generates an initial configuration based on a hexagonal packing.
    """
    centers = []
    # Approximate radius for 26 circles in unit square
    # 5x5 grid is 0.1. Hexagonal packing is denser.
    # Try r = 0.09 as a safe starting point
    r_start = 0.09
    diameter = 2 * r_start
    
    row = 0
    col = 0
    count = 0
    
    # Hexagonal lattice spacing
    # Horizontal distance: diameter
    # Vertical distance: sqrt(3)/2 * diameter
    
    y = r_start
    while count < n:
        x = r_start
        while count < n and x <= 1 - r_start:
            centers.append((x, y, r_start))
            count += 1
            x += diameter
        
        # Shift every other row
        if row % 2 == 1:
             # In hex packing, odd rows are shifted by r
             # But simple hex grid usually shifts by r (half diameter)
             pass 
             
        y += diameter * math.sqrt(3) / 2
        row += 1
        if row % 2 == 1:
             # Shift x for next row
             x_start_shifted = r_start + r_start # shift by radius? No, shift by half diameter = r
             # Actually standard hex: x coordinates are r, 3r, 5r... for row 0
             # and 2r, 4r... for row 1
             pass
             
    # The loop above is a bit simplistic. Let's generate points on a lattice.
    centers = []
    r = 0.085 # Start smaller to ensure fit
    
    # Number of rows roughly sqrt(N * sqrt(3)) ~ 6 or 7
    # Let's try to fill rows
    
    y = r
    row_idx = 0
    while len(centers) < n:
        # x coordinates for this row
        if row_idx % 2 == 0:
            xs = np.arange(r, 1 - r + 1e-9, 2 * r)
        else:
            xs = np.arange(r + r, 1 - r + 1e-9, 2 * r) # Shifted by r
            
        for x in xs:
            if len(centers) < n:
                # Check if x is valid (within [r, 1-r])
                if x >= r and x <= 1 - r:
                    centers.append((x, y, r))
        
        y += math.sqrt(3) * r
        row_idx += 1
        
    return np.array(centers).flatten()

def run_packing():
    """
    Main function to run the packing optimization.
    """
    n = 26
    
    # Initial guess
    # Try multiple starts to avoid local minima
    best_params = None
    best_score = -np.inf
    
    # Initialization 1: Hexagonal grid
    # We need to scale the grid to fit inside [0,1]x[0,1] with some margin
    # Let's create a dense grid and scale it down.
    
    # Create a raw hex grid
    points = []
    r_test = 1.0 # Radius in abstract space
    y = r_test
    row = 0
    while len(points) < n:
        if row % 2 == 0:
            xs = np.arange(r_test, 2 + r_test, 2 * r_test)
        else:
            xs = np.arange(2 * r_test, 2 + r_test, 2 * r_test)
        for x in xs:
            if len(points) < n:
                points.append([x, y])
        y += math.sqrt(3) * r_test
        row += 1
        
    points = np.array(points)
    
    # Center the points in [0, 2]x[0, 2] (since r=1, diameter=2)
    # Scale to fit in [0, 1] with some margin
    min_pt = np.min(points, axis=0)
    max_pt = np.max(points, axis=0)
    scale_x = 1.0 / (max_pt[0] - min_pt[0] + 0.1) # +0.1 for margin
    scale_y = 1.0 / (max_pt[1] - min_pt[1] + 0.1)
    
    # We want a single scale factor to preserve aspect ratio
    scale = min(scale_x, scale_y)
    
    # Center
    cx = (max_pt[0] + min_pt[0]) / 2
    cy = (max_pt[1] + min_pt[1]) / 2
    
    points_centered = (points - np.array([cx, cy])) * scale + np.array([0.5, 0.5])
    
    # Initial radii: estimate based on density
    # Area of square is 1. Area of 26 circles ~ 0.8?
    # 26 * pi * r^2 = 0.8 => r ~ 0.1
    r_init = 0.09
    
    params1 = np.zeros(n * 3)
    params1[0::3] = points_centered[:, 0]
    params1[1::3] = points_centered[:, 1]
    params1[2::3] = r_init
    
    # Initialization 2: Random valid packing
    np.random.seed(42)
    params2 = np.zeros(n * 3)
    # Place centers randomly in [0.2, 0.8] to ensure some space
    params2[0::3] = np.random.uniform(0.2, 0.8, n)
    params2[1::3] = np.random.uniform(0.2, 0.8, n)
    params2[2::3] = 0.02 # Small radii
    
    # Define constraints for SLSQP
    # Inequality constraints: g(x) >= 0
    cons = []
    
    # Overlap
    def get_overlap_func(p, n):
        return overlap_constraints(p, n)
    # SLSQP requires a callable that returns a numpy array
    # But we can't use closures. We need to pass n.
    # We can define a wrapper class or just use global n? 
    # The prompt says "No closures from function nesting".
    # I will define the constraints inside run_packing but pass them? 
    # Actually, scipy allows passing args. But the constraint function signature is fun(x).
    # I will use a global variable for N? Or just hardcode 26? 
    # N is 26. I will hardcode 26 in helper functions to avoid passing args if needed, 
    # but passing args is cleaner.
    # However, to strictly follow "no closures", I will define helpers that take params and n.
    # But scipy.optimize.minimize passes only x to constraints.
    # I will use `fun` with `args`.
    
    cons.append({
        'type': 'ineq',
        'fun': overlap_constraints,
        'args': (n,)
    })
    
    cons.append({
        'type': 'ineq',
        'fun': boundary_constraints,
        'args': (n,)
    })
    
    cons.append({
        'type': 'ineq',
        'fun': radius_constraints,
        'args': (n,)
    })
    
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius is 0.5)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    candidates = [params1, params2]
    
    for i, p0 in enumerate(candidates):
        try:
            res = opt.minimize(
                objective,
                p0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            if res.success and -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x.copy()
        except Exception as e:
            print(f"Optimization failed for start {i}: {e}")
            continue
            
    # If optimization didn't improve or failed, fallback to a simple valid grid
    if best_params is None:
        # Fallback: 5x5 grid + 1 small circle?
        # Just create a valid packing
        best_params = np.zeros(n * 3)
        idx = 0
        r = 0.09
        # 5x5 grid
        for r_idx in range(5):
            for c_idx in range(5):
                if idx < n:
                    x = 0.1 + c_idx * 0.2
                    y = 0.1 + r_idx * 0.2
                    best_params[3*idx] = x
                    best_params[3*idx+1] = y
                    best_params[3*idx+2] = r
                    idx += 1
        # Add 26th circle if needed (loop covers 25)
        if idx < n:
            # Place in center? No space. 
            # Shrink grid slightly
            r = 0.08
            for r_idx in range(5):
                for c_idx in range(5):
                     x = 0.12 + c_idx * 0.16 # Width 0.8
                     y = 0.12 + r_idx * 0.16
                     best_params[3*idx] = x
                     best_params[3*idx+1] = y
                     best_params[3*idx+2] = r
                     idx += 1
            # 26th circle in center hole?
            # Grid centers: 0.12, 0.28, 0.44, 0.60, 0.76
            # Hole at (0.20, 0.20) relative to 0.12?
            # Distance between centers 0.16. r=0.08. Touching.
            # Hole in middle of 4 circles. Dist to center sqrt(0.08^2+0.08^2) = 0.113.
            # r_hole = 0.113 - 0.08 = 0.033.
            if idx < n:
                best_params[3*idx] = 0.20 # approx center of 4 circles
                best_params[3*idx+1] = 0.20
                best_params[3*idx+2] = 0.03
                idx += 1

    # Extract results
    centers = best_params.reshape(-1, 3)[:, :2]
    radii = best_params.reshape(-1, 3)[:, 2]
    
    # Validate before returning to ensure robustness
    # (The validation function is provided, we can assume our constraints worked)
    # But let's clean up numerical errors
    # Clip radii to be safe
    radii = np.maximum(radii, 0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
