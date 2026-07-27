import numpy as np
import scipy.optimize as opt

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        return False

    if np.isnan(radii).any():
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                return False

    return True

def run_packing():
    N = 26
    
    # Initial configuration: Hexagonal lattice
    # We try to fit as many as possible in a hexagonal pattern
    # Approximate radius for 26 circles is around 0.101
    # Spacing in hexagonal packing: horizontal 2r, vertical sqrt(3)r
    
    r_est = 0.101
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    centers_init = []
    # Generate hexagonal points
    # Row 0
    # We need to fit 26 circles.
    # A pattern like 5-5-5-5-5-1? or 6-5-6-5-4?
    # Let's just generate a grid and pick the first 26, then optimize.
    # Or better, generate a dense hexagonal mesh and pick closest to center?
    
    # Let's try to construct a specific pattern known to be good or just a dense hex grid
    # 5 rows of 5 circles is 25. 
    # Let's try 6 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 1 circle?
    
    # Actually, optimization will handle the topology. We just need a feasible start.
    # Let's place them in a grid first to ensure feasibility, then let optimizer move them.
    # A 5x5 grid is feasible with r=0.1. 
    # We have 26 circles. 25 in grid, 1 extra.
    # Place the 26th in a corner or center?
    # Let's place 25 in a 5x5 grid and 1 at (0.5, 0.5) with small radius.
    # Then optimizer will expand.
    
    # Better start: Hexagonal packing of 26 equal circles.
    # Side length of triangle for n circles?
    # Let's just generate points on a hexagonal lattice and take 26 closest to (0.5, 0.5)
    # but bounded within [0,1].
    
    points = []
    y = r_est
    while y < 1 - r_est:
        x = r_est
        row_shift = 0.5 * dx if (y - r_est) // dy % 2 == 1 else 0 # Stagger rows
        while x < 1 - r_est:
            points.append([x + row_shift, y])
            x += dx
        y += dy
        
    # If we have too few points, fill with grid?
    # With r_est=0.101, dx=0.202, dy=0.175
    # y=0.101, 0.276, 0.451, 0.626, 0.801, 0.976(too high) -> 5 rows
    # Row 0 (y=0.101): x=0.101, 0.303, 0.505, 0.707, 0.909 -> 5 points
    # Row 1 (y=0.276): shift 0.101. x=0.202, 0.404, 0.606, 0.808 -> 4 points
    # Row 2 (y=0.451): 5 points
    # Row 3 (y=0.626): 4 points
    # Row 4 (y=0.801): 5 points
    # Total 5+4+5+4+5 = 23 points.
    # We need 26.
    
    # Let's use a denser grid start.
    # 6x5 grid = 30 points. Pick 26?
    # Or just 5x6 grid.
    # Let's create a 6x5 grid of points (30 points) and select 26 that are most "central" or just first 26.
    # Actually, just placing them in a 5x5 grid + 1 in the middle is a safe feasible start.
    
    grid_points = []
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    for y in ys:
        for x in xs:
            grid_points.append([x, y])
            
    # grid_points has 25.
    # Add one more in the center? (0.5, 0.5) is occupied.
    # Add one at (0.1, 0.1)? Occupied.
    # Let's shift the grid slightly or just place the 26th at (0.5, 0.5) with radius 0.01
    # But (0.5, 0.5) is a center of a circle in grid.
    # Let's place 26th at (0.05, 0.05) with small radius?
    # Or just duplicate a point with radius 0.
    
    # Let's try a hexagonal packing of 26 circles with smaller radius to fit, then expand.
    # Radius 0.08 fits easily.
    r_start = 0.08
    dx = 2 * r_start
    dy = np.sqrt(3) * r_start
    
    hex_points = []
    y = r_start
    while y + r_start <= 1.0 + 1e-9:
        x = r_start
        # Stagger
        row_idx = int(round((y - r_start) / dy))
        shift = row_idx * 0.5 * dx
        while x + r_start <= 1.0 + 1e-9:
            hex_points.append([x + shift, y])
            x += dx
        y += dy
    
    # If we have enough points, take first 26.
    if len(hex_points) >= 26:
        centers_init = hex_points[:26]
    else:
        # Fallback to grid + extra
        centers_init = grid_points.copy()
        centers_init.append([0.5, 0.5]) # Duplicate center, radius will adjust
        
    centers_init = np.array(centers_init)
    radii_init = np.full(N, r_start if len(hex_points) >= 26 else 0.01)
    
    # Ensure feasible start
    # If points are too close, radii must be small.
    # Let's compute min distance between points and set initial radii to min_dist/2
    min_dist = 1.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(centers_init[i] - centers_init[j])
            min_dist = min(min_dist, d)
    
    radii_init = np.full(N, min_dist / 2.0)
    
    # Optimization variables: x1, y1, r1, x2, y2, r2, ...
    # Total 3*N variables.
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max radius in square)
    bounds = []
    for _ in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
        
    # Objective: maximize sum of radii -> minimize -sum(r)
    def objective(vars):
        return -np.sum(vars[2::3])
        
    # Constraints
    # 1. Boundary: x >= r, x <= 1-r, y >= r, y <= 1-r
    #    => x - r >= 0, 1 - x - r >= 0, etc.
    # 2. Non-overlap: dist(i, j) >= r_i + r_j
    #    => (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    
    constraints = []
    
    # Boundary constraints
    for i in range(N):
        xi = 3*i
        yi = 3*i + 1
        ri = 3*i + 2
        
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[3*idx] - v[3*idx+2]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[3*idx] - v[3*idx+2]
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[3*idx+1] - v[3*idx+2]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[3*idx+1] - v[3*idx+2]
        })
        
    # Overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, ri = 3*i, 3*i+1, 3*i+2
            xj, yj, rj = 3*j, 3*j+1, 3*j+2
            
            def make_constraint(idx_i, idx_j):
                def fun(v):
                    xi_v = v[3*idx_i]
                    yi_v = v[3*idx_i+1]
                    ri_v = v[3*idx_i+2]
                    xj_v = v[3*idx_j]
                    yj_v = v[3*idx_j+1]
                    rj_v = v[3*idx_j+2]
                    
                    dist_sq = (xi_v - xj_v)**2 + (yi_v - yj_v)**2
                    sum_r = ri_v + rj_v
                    return dist_sq - sum_r**2
                return fun

            constraints.append({
                'type': 'ineq',
                'fun': make_constraint(i, j)
            })
            
    # Use SLSQP
    res = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
    
    if res.success:
        final_vars = res.x
        final_centers = np.zeros((N, 2))
        final_radii = np.zeros(N)
        for i in range(N):
            final_centers[i] = [final_vars[3*i], final_vars[3*i+1]]
            final_radii[i] = final_vars[3*i+2]
    else:
        # Fallback if optimization fails
        final_centers = centers_init
        final_radii = radii_init
        
    # Clean up small numerical errors
    final_radii = np.maximum(final_radii, 0)
    # Ensure centers are within bounds relative to radii (clamp if needed, though constraints should handle it)
    # But for safety
    for i in range(N):
        r = final_radii[i]
        cx, cy = final_centers[i]
        # Clamp center
        cx = np.clip(cx, r, 1-r)
        cy = np.clip(cy, r, 1-r)
        final_centers[i] = [cx, cy]

    # Final validation
    if not validate_packing(final_centers, final_radii):
        # If invalid, try to repair or return best effort
        # Simple repair: reduce radii slightly until valid
        scale = 1.0
        for _ in range(100):
            if validate_packing(final_centers, final_radii):
                break
            scale *= 0.99
            final_radii *= scale
    
    sum_radii = np.sum(final_radii)
    return final_centers, final_radii, sum_radii

# Execute
centers, radii, s_radii = run_packing()
print(f"Sum of radii: {s_radii}")