# sol_000167 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d2f1ce33) state=d804c6d5 sum of radii=2.615938 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    # Initialize centers and radii based on a hexagonal grid
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Hexagonal packing parameters
    # Approximate spacing for 26 circles in unit square
    # 5 rows, alternating 6 and 5 circles roughly
    rows = [6, 5, 6, 5, 4] # Total 26
    # Height calculation: (rows-1) * sqrt(3)/2 * 2r + 2r <= 1
    # Width calculation: max_circles * 2r <= 1 -> 6 * 2r <= 1 -> r <= 1/12 ~ 0.083
    # Let's aim for r ~ 0.09 initially
    r_init = 0.09
    y_curr = r_init
    idx = 0
    
    for row_idx, count in enumerate(rows):
        # Stagger rows by r_init
        x_start = r_init if row_idx % 2 == 0 else 2 * r_init
        for col in range(count):
            x = x_start + col * (2 * r_init)
            if x + r_init <= 1.0 + 1e-9:
                centers[idx] = [x, y_curr]
                radii[idx] = r_init
                idx += 1
        y_curr += np.sqrt(3) * r_init
        if idx >= n_circles:
            break
            
    # If we didn't fill 26 circles due to bounds, just reset to a random valid grid or grid
    if idx < n_circles:
        # Fallback to simple grid
        side = 6
        rem = n_circles % side
        row_counts = [side] * (n_circles // side)
        if rem > 0:
            row_counts.append(rem)
        
        y_step = 1.0 / (len(row_counts) + 1)
        current_y = y_step
        idx = 0
        for r_count in row_counts:
            x_step = 1.0 / (r_count + 1)
            for c in range(r_count):
                centers[idx] = [(c + 1) * x_step, current_y]
                radii[idx] = 0.08 # Initial small radius
                idx += 1
            current_y += y_step

    # Vector of all variables: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = centers[i, 0]
        x0[3*i + 1] = centers[i, 1]
        x0[3*i + 2] = radii[i]

    # Objective function: minimize negative sum of radii
    def objective(vars):
        return -np.sum(vars[2::3])

    # Constraints
    constraints = []
    
    # 1. Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    # Equivalent to: x - r >= 0, 1 - x - r >= 0, etc.
    for i in range(n_circles):
        idx_x = 3*i
        idx_y = 3*i + 1
        idx_r = 3*i + 2
        
        # x - r >= 0
        cons_x_lower = {
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_x] - v[idx_r]
        }
        constraints.append(cons_x_lower)
        
        # 1 - x - r >= 0
        cons_x_upper = {
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx_x] - v[idx_r]
        }
        constraints.append(cons_x_upper)
        
        # y - r >= 0
        cons_y_lower = {
            'type': 'ineq',
            'fun': lambda v, i=i: v[idx_y] - v[idx_r]
        }
        constraints.append(cons_y_lower)
        
        # 1 - y - r >= 0
        cons_y_upper = {
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[idx_y] - v[idx_r]
        }
        constraints.append(cons_y_upper)
        
        # r >= 0 (Handled by bounds usually, but good to be explicit if bounds don't cover)
        # We will use bounds for r >= 0

    # 2. Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            idx_xi, idx_yi, idx_ri = 3*i, 3*i + 1, 3*i + 2
            idx_xj, idx_yj, idx_rj = 3*j, 3*j + 1, 3*j + 2
            
            # To avoid lambda closure issues in loop, use a helper or direct args if supported
            # SLSQP fun arguments are passed. 
            # Since we can't use lambda with default args easily inside list append without defining function
            # We can define a class or use partial, but simplest is to define a specific function or use a list of functions.
            # However, standard practice in these prompts is to define constraints carefully.
            
            # We will add constraints later or define them dynamically.
            # Let's store the indices and create constraints in a loop carefully.
            pass

    # Re-defining constraints list properly to avoid closure issues
    constraints = []
    
    # Boundary constraints
    for i in range(n_circles):
        xi, yi, ri = 3*i, 3*i + 1, 3*i + 2
        
        # x >= r  => x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, x=xi, r=ri: v[x] - v[r]
        })
        # x <= 1-r => 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, x=xi, r=ri: 1.0 - v[x] - v[r]
        })
        # y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, y=yi, r=ri: v[y] - v[r]
        })
        # y <= 1-r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, y=yi, r=ri: 1.0 - v[y] - v[r]
        })

    # Overlap constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            xi, yi, ri = 3*i, 3*i + 1, 3*i + 2
            xj, yj, rj = 3*j, 3*j + 1, 3*j + 2
            
            # (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
            # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj: \
                    (v[xi]-v[xj])**2 + (v[yi]-v[yj])**2 - (v[ri]+v[rj])**2
            })

    # Bounds: radii >= 0, coords [0, 1]
    bounds = []
    for i in range(n_circles):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r (max possible radius is 0.5)

    # Optimization
    # Using SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 1000, 'ftol': 1e-9})

    if res.success:
        final_x = res.x
    else:
        # Fallback if optimization fails
        final_x = x0

    # Extract results
    final_centers = np.array([[final_x[3*i], final_x[3*i+1]] for i in range(n_circles)])
    final_radii = np.array([final_x[3*i+2] for i in range(n_circles)])
    
    # Ensure no negative radii due to numerical issues
    final_radii = np.maximum(final_radii, 0)
    
    total_sum = np.sum(final_radii)
    
    return final_centers, final_radii, total_sum

# Note: The function above might be slow or get stuck depending on starting point.
# To ensure a robust solution, we can refine the initial guess or run multiple times.
# However, for this prompt, we provide the single function.
# We will improve the initial guess logic slightly in the final code block to be more centered.

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    # Better initialization: Hexagonal packing centered in square
    # Estimate radius based on area or grid
    # 26 circles. 5x5 is 25. 6x5 is 30.
    # Try 6 rows of 5, but we only have 26. 
    # Let's place them in a grid and let optimizer adjust.
    
    # A good heuristic:
    # Try to fit in a rectangle.
    # 26 = 6 * 4 + 2?
    # Let's just place them on a hexagonal lattice within the square.
    
    r_guess = 0.1
    centers = []
    # Try to pack into rows
    # Row height = sqrt(3)*r
    # Row width for k circles = 2*r*k (approx, actually (2k-1)r + 2r? No, width 2rk if centers at r, 3r...)
    # Let's just use a simple loop
    
    # We want to maximize sum of radii, so we want them large.
    # Let's assume equal radii for initialization.
    
    # Let's try to find a valid configuration with r ~ 0.1
    # 5 rows. 
    # Row 1: 5 circles
    # Row 2: 5 circles
    # Row 3: 6 circles
    # Row 4: 5 circles
    # Row 5: 5 circles
    # Total 26.
    
    rows_config = [5, 5, 6, 5, 5]
    
    # Calculate required space for r=0.1
    # Width for 6 circles: 6 * 2 * 0.1 = 1.2 > 1. Too wide.
    # So we need smaller r or fewer circles per row.
    # For width 1, max circles with spacing 2r is floor(1/2r).
    # If r=0.1, max 5 circles.
    # So we can't have 6 circles in a row if they are size 0.1.
    # But they can be staggered.
    # In hexagonal packing, width of row with k circles is (k-1)*2r + 2r = 2kr?
    # Centers at r, 3r, ..., (2k-1)r.
    # Extent: [0, 2kr].
    # Yes. So for width 1, 2kr <= 1 => r <= 1/(2k).
    # For k=6, r <= 1/12 approx 0.0833.
    # For k=5, r <= 1/10 = 0.1.
    
    # If we use rows with 5 circles, r can be 0.1.
    # If we use rows with 6 circles, r is limited to 0.0833.
    # To maximize sum, we prefer 5-circle rows.
    # But we need 26 circles.
    # 5 rows of 5 = 25 circles. Sum radii = 2.5.
    # We need 1 more circle.
    # Where to put it?
    # In the gaps?
    # If we have a 5x5 grid of r=0.1, gaps are small.
    # But if we shift to hexagonal, we can fit more?
    # Or maybe we just allow one smaller circle?
    # But optimization will handle unequal radii.
    
    # Let's initialize with a grid of 25 circles of radius 0.1 and 1 circle of radius 0.05 somewhere?
    # Or just a loose grid.
    
    # Let's use a 6x5 grid (30 positions) but only populate 26.
    # And start with small radii, let them grow.
    
    # Grid approach:
    # 6 rows, 5 columns.
    # x coords: 0.1, 0.3, 0.5, 0.7, 0.9
    # y coords: 0.1, 0.25, 0.4, 0.55, 0.7, 0.85
    # This fits 30 circles. We pick first 26.
    
    x_coords = np.linspace(0.1, 0.9, 5)
    y_coords = np.linspace(0.1, 0.9, 6)
    
    init_centers = []
    for y in y_coords:
        for x in x_coords:
            init_centers.append([x, y])
            if len(init_centers) >= n_circles:
                break
        if len(init_centers) >= n_circles:
            break
            
    init_centers = np.array(init_centers[:n_circles])
    init_radii = np.full(n_circles, 0.08) # Start small
    
    # Vectorize
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = init_centers[i, 0]
        x0[3*i + 1] = init_centers[i, 1]
        x0[3*i + 2] = init_radii[i]
        
    # Define objective
    def objective(vars):
        return -np.sum(vars[2::3])
        
    # Constraints
    cons_list = []
    
    # Boundary
    for i in range(n_circles):
        xi, yi, ri = 3*i, 3*i + 1, 3*i + 2
        cons_list.append({'type': 'ineq', 'fun': lambda v, x=xi, r=ri: v[x] - v[r]})
        cons_list.append({'type': 'ineq', 'fun': lambda v, x=xi, r=ri: 1.0 - v[x] - v[r]})
        cons_list.append({'type': 'ineq', 'fun': lambda v, y=yi, r=ri: v[y] - v[r]})
        cons_list.append({'type': 'ineq', 'fun': lambda v, y=yi, r=ri: 1.0 - v[y] - v[r]})
        
    # Overlap
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            xi, yi, ri = 3*i, 3*i + 1, 3*i + 2
            xj, yj, rj = 3*j, 3*j + 1, 3*j + 2
            cons_list.append({
                'type': 'ineq', 
                'fun': lambda v, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj: \
                    (v[xi]-v[xj])**2 + (v[yi]-v[yj])**2 - (v[ri]+v[rj])**2
            })
            
    bounds = []
    for i in range(n_circles):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
        
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons_list, options={'maxiter': 2000})
    
    final_x = res.x
    final_centers = np.array([[final_x[3*i], final_x[3*i+1]] for i in range(n_circles)])
    final_radii = np.array([final_x[3*i+2] for i in range(n_circles)])
    final_radii = np.maximum(final_radii, 0)
    
    return final_centers, final_radii, np.sum(final_radii)
