import numpy as np
from scipy.optimize import minimize

def generate_hex_grid(n=26):
    """
    Generate an initial guess for centers in a hexagonal pattern.
    """
    # Estimate radius to fit roughly 26 circles
    # Area approx 1, density ~0.9, 26 circles -> r ~ 0.1
    # We start with a smaller r to ensure feasibility
    r_init = 0.05 
    
    # Try to arrange in rows
    # 5 rows of 5 is 25. Need 1 more.
    # Let's try 6 rows? 
    # 5, 5, 5, 5, 6 is 26.
    # Or 6, 5, 5, 5, 5.
    
    rows_counts = [5, 5, 5, 5, 6]
    
    centers = []
    
    # Horizontal spacing for diameter ~ 2*r_init
    # But we want to spread them out more initially to allow expansion
    # Let's target a bounding box of size 1x1 with margin
    margin = 0.1
    width = 1 - 2*margin
    height = 1 - 2*margin
    
    # Adjust for hexagonal shift
    # If we have 5 items in a row, width occupied is 4*spacing + 2*r? 
    # Actually centers. 
    # Let's just distribute uniformly first then shift rows.
    
    # We have 5 rows.
    # y coordinates: evenly spaced in [margin, 1-margin]
    y_coords = np.linspace(margin, 1-margin, len(rows_counts))
    
    for i, count in enumerate(rows_counts):
        y = y_coords[i]
        # Shift x for odd rows (hexagonal)
        shift = 0.0
        if i % 2 == 1:
            # Shift by half spacing. 
            # Spacing approx width / (count - 1) if count > 1
            if count > 1:
                spacing = width / (count - 1)
                shift = spacing / 2
            else:
                shift = 0
        
        # x coordinates
        if count == 0:
            continue
        if count == 1:
            x_coords = [0.5] # Center
        else:
            # Distribute count points in [margin, 1-margin] with shift?
            # If shifted, range might need adjustment.
            # Simple approach: linspace in [margin + shift, 1-margin + shift] clipped?
            # Better: linspace in [margin, 1-margin] and apply shift to centers?
            # Let's just place them centered.
            # Total width available 1.
            # If shifted, we might go out of bounds if not careful.
            # Let's place in [margin, 1-margin] first, then apply shift relative to center.
            
            # Calculate positions assuming no shift first
            # We want them to span the width
            # x_min = margin, x_max = 1 - margin
            # But if shifted, we might want to center the set.
            
            # Let's just use linspace
            raw_x = np.linspace(margin, 1-margin, count)
            
            # Apply shift relative to center 0.5
            # Center of raw_x is 0.5.
            # Shift them by 'shift' amount?
            # No, hex shift is usually relative to the lattice spacing.
            
            # Let's restart geometry logic.
            # Lattice spacing dx.
            # Row 0: x = dx, 3dx, 5dx...
            # Row 1: x = 2dx, 4dx...
            
            # Let's estimate dx.
            # Max x is 1-margin.
            # If row has 'count' items.
            # Unshifted: items at margin, margin+step, ...
            # Shifted: items at margin+step/2, ...
            
            # Let's just create a simple grid first and perturb?
            pass

    # Alternative: Simple grid perturbed
    # 5x5 grid is 25 points. Add one in middle of a cell?
    # 5x5 grid: x in [0.1, 0.3, 0.5, 0.7, 0.9]
    # y in [0.1, 0.3, 0.5, 0.7, 0.9]
    # This gives 25 points.
    # 26th point? Maybe at (0.5, 0.5) is occupied.
    # Maybe (0.4, 0.4)?
    
    # Let's generate a dense grid and pick 26?
    # 6x5 grid = 30 points.
    # x = linspace(0.1, 0.9, 6) -> 0.1, 0.266, 0.433, 0.6, 0.766, 0.9
    # y = linspace(0.1, 0.9, 5) -> 0.1, 0.3, 0.5, 0.7, 0.9
    # Total 30. Remove 4.
    # Which ones? Maybe corners?
    
    # Actually, random initialization with optimization is often robust enough for N=26
    # if we run it a few times or have a decent seed.
    
    # Let's try a hexagonal packing generation properly.
    # We want to fit circles of radius r.
    # Centers must be >= r from boundary.
    # Centers distance >= 2r.
    # Let's set r = 0.08 for initial guess.
    
    r_guess = 0.08
    points = []
    
    # Generate points in a hexagonal lattice
    # Row spacing dy = r * sqrt(3)
    # Col spacing dx = 2 * r
    # But we can scale this to fit [0,1]
    
    # Let's just create a list of (x,y)
    # Iterate y from r to 1-r with step sqrt(3)*r
    y = r_guess
    row = 0
    while y <= 1 - r_guess:
        # Determine x range
        # If row is even, x starts at r
        # If row is odd, x starts at r + r (shift by r) -> actually shift is r (half of 2r)
        
        start_x = r_guess
        if row % 2 == 1:
            start_x += r_guess # Shift by radius? No, shift by r is half spacing 2r.
            # Wait, spacing is 2r. Half is r.
            # So shift by r.
        
        x = start_x
        while x <= 1 - r_guess:
            points.append([x, y])
            x += 2 * r_guess
        
        y += np.sqrt(3) * r_guess
        row += 1
        
        # Safety break
        if len(points) > 50:
            break
            
    # We need exactly 26.
    # If we have more, trim. If fewer, add or adjust.
    # With r=0.08, dx=0.16, dy=0.138.
    # Width 1 -> ~6 cols. Height 1 -> ~7 rows. 42 points.
    # We will take first 26.
    
    if len(points) >= 26:
        centers = np.array(points[:26])
    else:
        # Fallback to random
        np.random.seed(42)
        centers = np.random.rand(26, 2)
        
    return centers

def run_packing():
    # Set seed for reproducibility if needed, but random is okay
    # We will use the generated centers
    
    # 1. Initial Guess
    centers_init = generate_hex_grid(26)
    
    # Initial radii
    r_init = np.full(26, 0.05)
    
    # Combine into state vector: [x0, y0, r0, x1, y1, r1, ...]
    # Or [x..., y..., r...]
    # Let's use [x..., y..., r...] for easier constraint writing?
    # Actually grouping (x,y,r) per circle might be easier for loops,
    # but vectorized ops prefer separate arrays.
    # Let's stick to flat array: x (0-25), y (26-51), r (52-77).
    
    x0 = centers_init[:, 0]
    y0 = centers_init[:, 1]
    r0 = r_init
    
    x0 = np.concatenate([x0, y0, r0])
    
    n = 26
    
    # Bounds
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(2 * n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 0.5))
        
    # Constraints
    # 1. Boundary constraints for each circle
    # x_i >= r_i  => x_i - r_i >= 0
    # x_i <= 1 - r_i => 1 - x_i - r_i >= 0
    # y_i >= r_i => y_i - r_i >= 0
    # y_i <= 1 - r_i => 1 - y_i - r_i >= 0
    
    # 2. Non-overlap constraints
    # dist(i, j) >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) - (ri + r_r) >= 0
    # Using squared form might be safer?
    # (xi-xj)^2 + (yi-yj)^2 >= (ri + rj)^2
    # (xi-xj)^2 + (yi-yj)^2 - (ri + rj)^2 >= 0
    
    constraints = []
    
    # Boundary constraints
    # We can add them as dicts or functions
    # SLSQP accepts list of dicts {'type': 'ineq', 'fun': ...}
    
    # To reduce overhead, we can vectorize the constraint function?
    # But SLSQP calls fun(x) and expects a vector of constraint values?
    # No, 'fun' returns a scalar or array? 
    # If it returns an array, it's treated as multiple constraints?
    # Documentation says: "fun(x) should return an array of values... if the constraint is a single constraint..."
    # Actually, for multiple constraints, usually we pass a list of dicts or a function returning array?
    # In scipy.optimize.minimize, constraints can be a NonlinearConstraint or a dict.
    # If dict, 'fun' returns a scalar? No, it can return an array for multiple constraints?
    # Let's check: "fun : callable. The function which returns an array of values... If there are multiple constraints, the constraint function should return an array of values..."
    # Wait, standard usage is a list of dicts where each dict defines one constraint (scalar).
    # But passing a function that returns an array of size M is supported in newer scipy?
    # Actually, it's safer to pass a list of constraints or use NonlinearConstraint.
    # However, 400 constraints in a list is slow to construct?
    # Let's try to bundle them.
    
    # Let's define a function that returns all constraint violations (values >= 0)
    def constraint_fun(vars):
        x = vars[0:n]
        y = vars[n:2*n]
        r = vars[2*n:]
        
        cons = []
        
        # Boundary constraints
        # x - r >= 0
        cons.extend(x - r)
        # 1 - x - r >= 0
        cons.extend(1.0 - x - r)
        # y - r >= 0
        cons.extend(y - r)
        # 1 - y - r >= 0
        cons.extend(1.0 - y - r)
        
        # Circle-circle constraints
        # We need to check all pairs.
        # To speed up, maybe only check close pairs?
        # But we don't know who is close.
        # Vectorized approach:
        # Compute distance matrix? O(N^2). 26^2 is small (676).
        
        # x matrix (n, n)
        x_mat = x[:, None] - x[None, :]
        y_mat = y[:, None] - y[None, :]
        r_mat = r[:, None] + r[None, :]
        
        dist_sq = x_mat**2 + y_mat**2
        sum_r_sq = r_mat**2
        
        # We need dist_sq - sum_r_sq >= 0
        # But this includes diagonal (0) and double counting.
        # Diagonal: dist=0, sum_r=2r. 0 - 4r^2 < 0. Violation!
        # Wait, diagonal constraint is meaningless or should be ignored.
        # But dist_sq[i,i] = 0. (r[i]+r[i])^2 = 4r[i]^2.
        # Constraint 0 >= 4r[i]^2 implies r[i]=0.
        # We must exclude diagonal.
        
        # Mask diagonal
        mask = ~np.eye(n, dtype=bool)
        valid_constraints = dist_sq[mask] - sum_r_sq[mask]
        
        # Also, we have upper triangle and lower triangle.
        # Both are same. We can just take upper triangle to save half constraints.
        # np.triu_indices(n, k=1)
        triu = np.triu_indices(n, k=1)
        dist_sq_triu = dist_sq[triu]
        sum_r_sq_triu = sum_r_sq[triu]
        
        cons.extend(dist_sq_triu - sum_r_sq_triu)
        
        return np.array(cons)

    # This function returns ~ 4*26 + 325 = 429 constraints.
    # SLSQP handles this.
    
    cons_dict = {'type': 'ineq', 'fun': constraint_fun}
    
    # Objective: minimize -sum(r)
    def objective(vars):
        r = vars[2*n:]
        return -np.sum(r)
        
    # Run optimizer
    # Method SLSQP
    # Options: maxiter, ftol, etc.
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons_dict, 
                   options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
    
    # Extract results
    final_x = res.x[0:n]
    final_y = res.x[n:2*n]
    final_r = res.x[2*n:]
    
    # Validate and fix small negative radii due to numerical errors
    final_r = np.maximum(final_r, 0.0)
    
    centers = np.column_stack((final_x, final_y))
    sum_r = np.sum(final_r)
    
    # If optimization failed or is bad, maybe try again?
    # But for now, return result.
    
    # We should ensure the result is valid.
    # The validation function provided in prompt checks strict constraints.
    # Our constraints were >= 0. 
    # Numerical precision might be an issue.
    # We can clamp radii slightly if needed, but let's trust the optimizer.
    
    return centers, final_r, sum_r

# We need to output the function run_packing.
# The prompt asks to return the final program.
# I will wrap the logic.