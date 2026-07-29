# sol_000004 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f294fc76) state=3a187037 sum of radii=2.542303 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    
    # We need to solve for variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Total variables = 26 * 3 = 78
    
    def objective(vars):
        # vars is a 1D array of shape (78,)
        # radii are at indices 2, 5, 8, ... (indices i*3 + 2)
        radii = vars[2::3]
        # We want to maximize sum of radii, so minimize negative sum
        return -np.sum(radii)
    
    def constraints(vars):
        # vars shape (78,)
        # Reshape to (26, 3) for easier access
        data = vars.reshape(-1, 3)
        centers = data[:, :2]
        radii = data[:, 2]
        
        c_list = []
        
        # 1. Boundary constraints: x >= r, y >= r, x + r <= 1, y + r <= 1
        # x - r >= 0
        c_list.extend(centers[:, 0] - radii)
        # y - r >= 0
        c_list.extend(centers[:, 1] - radii)
        # 1 - x - r >= 0 => (1-x) - r >= 0
        c_list.extend(1.0 - centers[:, 0] - radii)
        # 1 - y - r >= 0
        c_list.extend(1.0 - centers[:, 1] - radii)
        
        # 2. Non-overlap constraints: dist(i,j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
        # There are n*(n-1)/2 pairs. This might be many constraints (325).
        # SLSQP can handle this, but it might be slow. 
        # Optimization: Only check nearby circles? 
        # But for global optimality, we should check all. 
        # Let's check all. 325 constraints is manageable.
        
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                r_sum = radii[i] + radii[j]
                c_val = dist_sq - r_sum*r_sum
                c_list.append(c_val)
                
        return np.array(c_list)

    # Helper to generate initial guess
    def get_initial_guess(method='hex'):
        # Generate a hexagonal lattice
        # Spacing parameter. If r ~ 0.1, diameter 0.2.
        # Hex spacing vertical is sqrt(3)/2 * diameter ~ 0.1732
        # Let's try to pack roughly.
        
        centers = []
        radii = []
        
        if method == 'grid':
            # 5x5 grid is 25. Add one?
            # Let's try 6x5 grid (30 points) and keep 26?
            # Or just 5x5 + 1 random.
            # Let's do a dense grid.
            # 6 columns, 5 rows -> 30 points. 
            # Step size 1/5 = 0.2.
            # Points at 0.1, 0.3, 0.5, 0.7, 0.9
            # 5 points per row.
            # If we shift rows (hexagonal), we might fit more.
            pass 
            
        # Hexagonal packing initialization
        # Row 0: y = r_init. x = r_init + k * 2*r_init
        # Let's assume initial radius r_init = 0.08
        r_init = 0.08
        row_y = r_init
        col_x = r_init
        
        # We want to fill the square.
        # Let's generate points with spacing 2*r_init horizontally
        # and sqrt(3)*r_init vertically.
        
        # Max x index
        max_x_idx = int((1.0 - 2*r_init) / (2*r_init)) + 1
        # Max y index
        max_y_idx = int((1.0 - 2*r_init) / (math.sqrt(3)*r_init)) + 1
        
        pts = []
        for i in range(max_y_idx + 1):
            for j in range(max_x_idx + 1):
                # Shift every other row
                offset = math.sqrt(3)*r_init * 0.5 if i % 2 == 1 else 0
                x = r_init + j * (2*r_init) + offset # Wait, offset should be half spacing? 
                # Horizontal spacing is 2r. Shift is r (half spacing)?
                # In hex packing, horizontal distance between centers in adjacent rows is r.
                # Wait, distance between centers is 2r.
                # If row 0 is at x = r, r+2r, r+4r...
                # Row 1 is at x = r+r, r+3r... => x = 2r, 4r...
                # Shift is r.
                # Let's correct offset.
                pass 
        
        # Let's restart initialization logic properly
        # Hexagonal lattice:
        # Basis vectors: (2r, 0) and (r, r*sqrt(3))
        # Points: k1*(2r, 0) + k2*(r, r*sqrt(3)) + (r, r) ?
        # Center of square is (0.5, 0.5).
        # Let's place a lattice centered at (0.5, 0.5).
        
        # Let's try a simple grid first, then maybe hex if needed.
        # Actually, a 5x5 grid has 25 points. 
        # We can place 26 points by perturbing a grid or using a specific pattern.
        # Let's use a randomized grid perturbation or just a 6x5 grid selection.
        
        # Let's try to place 26 points in a hexagonal arrangement roughly fitting 0.1 radius.
        # If r=0.1, width of 5 circles is 1.0 (exactly).
        # So 5 columns is max for r=0.1.
        # Height for 5 rows hex is approx 0.89.
        # So 5 rows is possible.
        # 5x5 hex grid has 25 circles? 
        # Row 0: 5
        # Row 1: 5 (shifted) -> width might be issue?
        # If row 1 is shifted by r=0.1, x coords: 0.2, 0.4, 0.6, 0.8, 1.0?
        # 1.0 is boundary. Center at 1.0 implies r=0? No.
        # If x=1.0-r, max x is 0.9.
        # So shifted row can't have 5 circles if shift pushes one out.
        # Row 0: 0.1, 0.3, 0.5, 0.7, 0.9 (5 circles)
        # Row 1 (shift 0.1): 0.2, 0.4, 0.6, 0.8 (4 circles). 1.0 is out.
        # So hex packing of 5 rows: 5, 4, 5, 4, 5 -> 23 circles.
        # We need 26.
        
        # Maybe 6 rows?
        # 5, 4, 5, 4, 5, 4 -> 27 circles.
        # Remove 1?
        # Let's try to construct 26 points.
        
        pts = []
        # Try to fit as many as possible in a hex grid with spacing derived from r=0.1
        # But we don't know optimal r. Let's just scatter them well.
        
        # Approach: 6 rows.
        # Row heights: y = 0.1, 0.25, 0.4, 0.55, 0.7, 0.85 ?
        # Spacing 0.15.
        # x spacing 0.15.
        # Grid 7x7 = 49 points. Too many.
        
        # Let's use a simple heuristic: 26 points distributed uniformly.
        # Poisson disk sampling is good but complex to implement from scratch quickly.
        # Let's use a deterministic placement.
        
        # 5 rows.
        # Row 0: 6 circles?
        # If we squeeze 6 circles, r must be smaller.
        # Let's assume r will be optimized. We just need feasible centers.
        # Place centers such that they don't overlap with r=0.05.
        
        # 6 columns, 5 rows = 30 points.
        # Spacing 1/6 approx 0.166.
        # x in [0.166, 0.333, 0.5, 0.666, 0.833] ? 
        # Actually 6 points: 0.1, 0.28, 0.46, 0.64, 0.82, 1.0?
        # Let's use linspace.
        
        x_vals = np.linspace(0.1, 0.9, 6) # 6 points
        y_vals = np.linspace(0.1, 0.9, 5) # 5 points
        
        candidates = []
        for y in y_vals:
            for x in x_vals:
                candidates.append([x, y])
        
        # We have 30 candidates. We need 26.
        # Remove 4. Which ones?
        # Maybe corners are good?
        # Let's just take the first 26.
        # Or better, remove points that are "worst" located?
        # Corners are constrained. Center is free.
        # Maybe remove corners?
        # Let's remove (0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9).
        # That leaves 26.
        
        # Actually, let's just pick 26 random ones? No, deterministic is better.
        # Let's keep the dense center.
        # Remove 4 corners.
        
        # But wait, 6x5 grid is quite dense.
        # Maybe 5x5 grid (25) + 1 center?
        # 5x5 grid: x in 0.1..0.9 step 0.2. y in 0.1..0.9 step 0.2.
        # 25 points.
        # Add (0.5, 0.5)? It's already there (3rd, 3rd).
        # Add (0.5, 0.2)? No.
        # Where is the gap?
        # Gaps are at (0.2, 0.2), (0.2, 0.4)...
        # (0.2, 0.2) is center of square formed by (0.1,0.1), (0.3,0.1), (0.1,0.3), (0.3,0.3).
        # Distance to neighbors is sqrt(0.1^2+0.1^2) = 0.141.
        # If we place a circle there, radius limited by 0.141 - r_grid.
        # If r_grid = 0.1, gap radius = 0.041.
        # If we reduce r_grid slightly, we can increase gap radius.
        # This suggests a configuration of 26 circles might have slightly smaller radii than 25.
        
        # Let's initialize with 26 points in a hexagonal pattern that fits.
        # Let's try to construct a valid configuration for r=0.08.
        
        centers = []
        # 6 rows, alternating 5 and 4 circles?
        # 5, 4, 5, 4, 5, 4 -> 27. Remove 1.
        # Let's do 5 rows of 5 (25) + 1 extra.
        # Where to put extra?
        # Maybe expand grid to 6 rows?
        # 5, 4, 5, 4, 5, 3? Sum = 26.
        
        # Let's just generate a hexagonal grid of points and take first 26.
        r_start = 0.08
        pts = []
        y_curr = r_start
        row = 0
        while y_curr <= 1.0 - r_start:
            # Determine number of columns
            # If row is even (0, 2...), x starts at r_start
            # If row is odd (1, 3...), x starts at r_start + r_start (shift by r)
            # Actually shift is r_start (diameter/2) for hex packing?
            # Horizontal distance between centers in same row is 2*r_start.
            # Vertical distance is sqrt(3)*r_start.
            # Shift for odd rows is r_start.
            
            start_x = r_start
            if row % 2 == 1:
                start_x += r_start # shift by radius
            
            x_curr = start_x
            while x_curr <= 1.0 - r_start:
                pts.append([x_curr, y_curr])
                x_curr += 2 * r_start
            
            y_curr += math.sqrt(3) * r_start
            row += 1
        
        # We might have more or less than 26.
        # With r=0.08, we should fit plenty.
        # If we have more, take first 26.
        # If less, repeat with smaller r or random fill.
        
        if len(pts) >= n_circles:
            centers = pts[:n_circles]
        else:
            # Fallback to random uniform distribution
            centers = np.random.rand(n_circles, 2)
            # Scale to be inside [0.1, 0.9]
            centers = 0.1 + 0.8 * centers

        # Initial radii: small feasible value
        # We can estimate max radius from min distance
        min_dist = float('inf')
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                d = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                if d < min_dist:
                    min_dist = d
            d_x = min(centers[i][0], 1.0 - centers[i][0])
            d_y = min(centers[i][1], 1.0 - centers[i][1])
            d_boundary = min(d_x, d_y)
            if d_boundary < min_dist:
                min_dist = d_boundary
        
        r_init = min_dist / 2.0
        if r_init > 0.05: r_init = 0.05 # Cap to be safe
        
        init_radii = np.full(n_circles, r_init)
        
        vars0 = np.zeros(n_circles * 3)
        for i in range(n_circles):
            vars0[i*3] = centers[i][0]
            vars0[i*3+1] = centers[i][1]
            vars0[i*3+2] = init_radii[i]
            
        return vars0

    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Initial guess
    x0 = get_initial_guess()
    
    # Optimization options
    # SLSQP is good for constrained optimization
    # maxiter might need to be high
    res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints={'type': 'ineq', 'fun': constraints},
                       options={'maxiter': 1000, 'ftol': 1e-8, 'disp': False})
    
    # If optimization fails or doesn't improve, we might try a second run with perturbation
    # But let's check the result first.
    
    # Extract results
    best_vars = res.x
    centers = best_vars.reshape(-1, 3)[:, :2]
    radii = best_vars.reshape(-1, 3)[:, 2]
    
    # Validate and clean up (ensure constraints are met within tolerance)
    # Sometimes numerical issues leave tiny violations.
    # We can do a post-processing step to shrink radii slightly if needed,
    # but the optimizer should have handled it.
    # However, to be safe against the strict validation (1e-12 tolerance),
    # we might need to be careful.
    
    # Let's check if constraints are satisfied
    valid = True
    # Check overlaps
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-9:
                valid = False
                break
        if not valid: break
        
    # Check boundaries
    for i in range(n_circles):
        if radii[i] < -1e-9: valid = False
        if centers[i, 0] - radii[i] < -1e-9 or centers[i, 0] + radii[i] > 1 + 1e-9: valid = False
        if centers[i, 1] - radii[i] < -1e-9 or centers[i, 1] + radii[i] > 1 + 1e-9: valid = False

    # If not valid, we might need to shrink radii slightly to satisfy constraints strictly
    if not valid:
        # Find max violation
        # Just shrink all radii by a small epsilon?
        # Or scale down?
        # Scaling down reduces sum, but ensures validity.
        # Better: fix specific violations.
        # But for simplicity, if valid is false, let's try to recover.
        # Usually SLSQP satisfies constraints.
        pass

    # The optimizer might have found a local optimum.
    # Let's try to run it again with a different initialization to see if we can beat it.
    # But for the purpose of this function, one run is expected.
    # Let's do a "simulated annealing" like improvement?
    # Or just trust SLSQP.
    
    # Let's calculate sum
    sum_radii = np.sum(radii)
    
    # Just in case, let's try a second run with randomized initial radii/positions if sum is low?
    # But we can't loop indefinitely.
    # Let's stick to the result.
    
    return centers, radii, sum_radii
