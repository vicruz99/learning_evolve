# sol_000041 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 16d9c155) state=1ca6e21b sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def run_packing():
    """
    Returns (centers, radii, sum_radii) for a packing of 26 circles in a unit square.
    """
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We try to arrange 26 circles in a hexagonal pattern.
    # A hexagonal packing has rows shifted by r horizontally and spaced by r*sqrt(3) vertically.
    # Let's assume an initial radius r0 = 0.09 (safe guess)
    r0 = 0.09
    
    # Determine rows and columns
    # For 26 circles, maybe 6 rows.
    # Row counts: 5, 5, 5, 5, 5, 1? Or 5, 5, 5, 5, 4, 2?
    # Let's try to fill rows greedily.
    # Max circles in a row with r=0.09: width 1. 2r = 0.18. 
    # (1 - 2r) / 2r + 1 = (0.82)/0.18 + 1 approx 5.5. So 5 or 6.
    # Let's try 6 rows with 5 circles, and add 1 extra? No, 6*5 = 30.
    # We need 26.
    # Configuration: 5, 5, 5, 5, 4, 2 = 26.
    # Or 5, 5, 5, 5, 5, 1 (centered).
    
    # Let's generate a dense hexagonal grid and pick the first 26 points that fit.
    centers_init = []
    r_init = 0.09
    
    # Hexagonal packing coordinates
    # y coordinates: r, r + r*sqrt(3), r + 2*r*sqrt(3), ...
    # x coordinates: r, r + 2r, r + 4r... for even rows
    #                 r + r, r + 3r... for odd rows (shifted by r)
    
    sqrt3 = np.sqrt(3)
    row_idx = 0
    count = 0
    
    while count < n:
        y = r_init + row_idx * r_init * sqrt3
        if y + r_init > 1.0:
            break
            
        # Determine x coordinates for this row
        # If row_idx is even (0, 2, ...), start at r_init
        # If row_idx is odd (1, 3, ...), start at 2*r_init (shifted by r)
        
        if row_idx % 2 == 0:
            x_start = r_init
        else:
            x_start = 2 * r_init
            
        x = x_start
        while x + r_init <= 1.0:
            centers_init.append([x, y])
            count += 1
            if count == n:
                break
            x += 2 * r_init
        row_idx += 1
        
    # If we didn't get 26, adjust or just use what we have and add some
    # With r=0.09, we should get enough.
    # Let's ensure we have exactly 26.
    if len(centers_init) < n:
        # Fallback to random or grid
        # Just fill remaining with small circles in gaps? 
        # Or simpler: Grid
        step = 1.0 / 6.0 # 0.166
        for i in range(n):
            row = i // 5
            col = i % 5
            centers_init.append([col * 0.2 + 0.1, row * 0.2 + 0.1])
        centers_init = centers_init[:n]
        
    centers_init = np.array(centers_init[:n])
    
    # 2. Optimization
    # Variables: x1, y1, r1, x2, y2, r2, ...
    # Order: x[0], y[0], r[0], x[1], y[1], r[1] ...
    # Total variables: 26 * 3 = 78
    
    # Initial values
    x0 = np.zeros(78)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i + 1] = centers_init[i, 1]
        x0[3*i + 2] = r_init
    
    # Bounds: x in [0, 1], y in [0, 1], r in [0, 0.5]
    # Actually r upper bound can be 0.5, but practically smaller.
    bnds = []
    for _ in range(n):
        bnds.append((0, 1)) # x
        bnds.append((0, 1)) # y
        bnds.append((0, 0.5)) # r

    # Objective: Maximize sum of radii -> Minimize -sum(r)
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)

    # Constraints
    # 1. Boundary constraints:
    # x >= r, x <= 1-r  => r <= x <= 1-r
    # y >= r, y <= 1-r  => r <= y <= 1-r
    # This is equivalent to x - r >= 0, 1 - x - r >= 0, etc.
    # Nonlinear constraints in scipy: fun(vars) >= 0
    
    def boundary_constraints(vars):
        cons = []
        for i in range(n):
            x = vars[3*i]
            y = vars[3*i + 1]
            r = vars[3*i + 2]
            # x >= r => x - r >= 0
            cons.append(x - r)
            # 1 - x >= r => 1 - x - r >= 0
            cons.append(1 - x - r)
            # y >= r
            cons.append(y - r)
            # 1 - y >= r
            cons.append(1 - y - r)
        return np.array(cons)

    # 2. Non-overlap constraints:
    # dist(i, j) >= r_i + r_j
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    def overlap_constraints(vars):
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                xi = vars[3*i]
                yi = vars[3*i + 1]
                ri = vars[3*i + 2]
                
                xj = vars[3*j]
                yj = vars[3*j + 1]
                rj = vars[3*j + 2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                rad_sum = ri + rj
                cons.append(dist_sq - rad_sum**2)
        return np.array(cons)

    # Using NonlinearConstraint for efficiency if possible, but function returning array is fine for SLSQP?
    # SLSQP accepts a list of constraint dicts or a single constraint dict with fun returning array?
    # SLSQP supports dict with 'fun' and 'ineq' or 'eq'. 'fun' must return scalar or array?
    # If array, it treats each element as a constraint? 
    # Actually, for SLSQP, 'fun' in dict usually returns scalar. For multiple, list of dicts.
    # But creating 325 dicts is slow.
    # Let's use the function that returns an array and pass it as a constraint? 
    # Wait, SLSQP in scipy.optimize.minimize:
    # constraints can be a NonlinearConstraint object.
    
    # Boundary constraints count: 4 * 26 = 104
    n_boundary_cons = 4 * n
    bound_lb = np.zeros(n_boundary_cons)
    bound_ub = np.inf * np.ones(n_boundary_cons)
    
    # Overlap constraints count: 26*25/2 = 325
    n_overlap_cons = n * (n - 1) // 2
    overlap_lb = np.zeros(n_overlap_cons)
    overlap_ub = np.inf * np.ones(n_overlap_cons)

    # Define constraint objects
    con_boundary = NonlinearConstraint(boundary_constraints, bound_lb, bound_ub)
    con_overlap = NonlinearConstraint(overlap_constraints, overlap_lb, overlap_ub)

    # Run optimization
    # SLSQP might struggle with 325 constraints. 
    # Let's try to run it.
    
    # To speed up, maybe reduce precision or use fewer iterations?
    # Or use a different method? 
    # 'SLSQP' is robust.
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bnds, 
                       constraints=[con_boundary, con_overlap],
                       options={'maxiter': 200, 'ftol': 1e-9})
        vars_opt = res.x
    except Exception as e:
        # Fallback if optimization fails
        # Just return initial guess with radii adjusted to fit?
        vars_opt = x0 

    # Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = vars_opt[3*i]
        centers[i, 1] = vars_opt[3*i + 1]
        radii[i] = vars_opt[3*i + 2]

    sum_radii = np.sum(radii)
    
    # Post-processing: ensure validity (handle numerical errors)
    # Clip radii to non-negative
    radii = np.maximum(radii, 0)
    # Clip centers to [0,1]
    centers = np.clip(centers, 0, 1)
    
    # Check if valid
    # If not valid, try to shrink radii slightly to satisfy constraints
    # But with optimization, it should be valid.
    # Just in case, let's verify and fix if needed.
    
    # Verification loop
    valid = False
    for _ in range(100):
        # Check overlaps
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-9:
                    # Shrink radii slightly
                    factor = 0.99
                    radii *= factor
                    overlap_found = True
                    break
            if overlap_found:
                break
        if not overlap_found:
            # Check boundaries
            boundary_violation = False
            for i in range(n):
                if centers[i, 0] < radii[i] - 1e-9 or centers[i, 0] > 1 - radii[i] + 1e-9:
                    # Adjust center or shrink
                    radii[i] *= 0.95
                    boundary_violation = True
                if centers[i, 1] < radii[i] - 1e-9 or centers[i, 1] > 1 - radii[i] + 1e-9:
                    radii[i] *= 0.95
                    boundary_violation = True
            if not boundary_violation:
                valid = True
                break
    
    # Recalculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
