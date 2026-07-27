import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    
    # 1. Hexagonal Grid Initialization
    # We create a staggered grid to maximize initial density.
    # 6 rows: 5 circles in rows 1, 2, 3, 4, 5, and 1 circle in row 6 to total 26.
    # Alternatively, 5 rows of 5 and 1 extra, but 6x5 staggered is a robust base.
    
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    idx = 0
    
    # Row configurations: 5 rows of 5 circles
    row_counts = [5, 5, 5, 5, 5]
    
    # Adjust radii and spacing to fit in [0,1]x[0,1]
    # Approximate r for 5 rows hexagonal:
    # Height H = 2r + 4 * (r * sqrt(3)) = r * (2 + 4*sqrt(3)) approx 8.928r
    # r approx 1/9
    # Width W = 10r approx 1.11 (if 5 circles side-by-side)
    # We scale the grid to fit.
    
    temp_r = 0.09 # Initial guess
    vert_spacing = temp_r * np.sqrt(3)
    horiz_spacing = 2 * temp_r
    
    # Calculate bounds for centering
    # y centers
    y_coords = []
    y_step = vert_spacing
    # 5 rows -> 4 steps between centers
    # Total height of centers span = 4 * y_step
    # We want this span + 2r to fit in 1
    # Actually, we will just normalize the grid later.
    
    raw_centers = []
    y_base = 0
    for row_idx in range(5):
        for col_idx in range(5):
            # Hexagonal shift: odd rows shifted by r (horiz_spacing / 2)
            x_offset = 0
            if row_idx % 2 == 1:
                x_offset = horiz_spacing / 2
            
            x = col_idx * horiz_spacing + x_offset
            y = row_idx * y_step
            raw_centers.append([x, y])
            
    # We have 25 centers. We need 1 more.
    # We can place the 26th circle in the "center" of the square if space permits,
    # or just add it to the grid logic. 
    # Let's stick to 5 rows of 5, and add the 26th circle at the geometric center (0.5, 0.5)
    # with a small initial radius.
    raw_centers.append([0.5, 0.5])
    
    raw_centers = np.array(raw_centers)
    
    # Normalize to fit in [0,1]x[0,1] with a margin
    # We scale so that the bounding box of centers is roughly within [r, 1-r]
    # But for initialization, let's just scale to fit in [0,1]
    x_min, y_min = raw_centers.min(axis=0)
    x_max, y_max = raw_centers.max(axis=0)
    
    x_span = x_max - x_min
    y_span = y_max - y_min
    
    # Scale to 80% of the unit square to leave room for radii
    scale = 0.8 / max(x_span, y_span, 1e-9)
    
    scaled_centers = raw_centers * scale
    
    # Center the configuration in the unit square
    c_x = scaled_centers[:, 0].mean()
    c_y = scaled_centers[:, 1].mean()
    scaled_centers[:, 0] -= c_x - 0.5
    scaled_centers[:, 1] -= c_y - 0.5
    
    # Initial radii: small value to ensure valid start
    initial_radii = np.ones(n) * 0.01
    
    # 2. Optimization
    # Variables: [x0, y0, r0, x1, y1, r1, ...]
    # Total 3 * n variables
    x0 = np.concatenate([scaled_centers.flatten(), initial_radii])
    
    def objective(vars_flat):
        radii = vars_flat[2*n:]
        return -np.sum(radii) # Minimize negative sum
    
    def boundary_constraints(vars_flat):
        centers = vars_flat[:2*n].reshape(n, 2)
        radii = vars_flat[2*n:]
        con = []
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # x - r >= 0
            con.append(x - r)
            # 1 - x - r >= 0 => x + r <= 1
            con.append(1.0 - x - r)
            # y - r >= 0
            con.append(y - r)
            # 1 - y - r >= 0 => y + r <= 1
            con.append(1.0 - y - r)
        return np.array(con)

    def overlap_constraints(vars_flat):
        centers = vars_flat[:2*n].reshape(n, 2)
        radii = vars_flat[2*n:]
        con = []
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.sum((centers[i] - centers[j])**2)
                # dist >= r_i + r_j  =>  dist^2 >= (r_i + r_j)^2
                # But square root is safer for gradients? No, dist - (r_i + r_j) >= 0 is fine.
                # dist = sqrt(dist_sq)
                dist = np.sqrt(dist_sq)
                con.append(dist - (radii[i] + radii[j]))
        return np.array(con)

    # Constraints setup
    # We need to pass constraints to SLSQP
    # SLSQP expects a list of dictionaries
    
    cons = []
    
    # Boundary constraints
    cons.append({
        'type': 'ineq',
        'fun': boundary_constraints
    })
    
    # Overlap constraints
    cons.append({
        'type': 'ineq',
        'fun': overlap_constraints
    })

    # Bounds: x, y in [0, 1], r >= 0
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, None)) # r
    
    # Run optimization
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                      options={'maxiter': 1000, 'ftol': 1e-9})
    
    final_vars = result.x
    final_centers = final_vars[:2*n].reshape(n, 2)
    final_radii = final_vars[2*n:]
    
    return final_centers, final_radii, np.sum(final_radii)