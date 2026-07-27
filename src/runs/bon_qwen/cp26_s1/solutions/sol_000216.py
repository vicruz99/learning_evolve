# sol_000216 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5b6844e7) state=1192ffb2 sum of radii=2.310015 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # ---------------------------------------------------------
    # 1. Initialization: Hexagonal Grid Layout
    # ---------------------------------------------------------
    # We want to place 26 circles. A hexagonal packing is denser than square.
    # We generate a set of candidate points in a hexagonal lattice and pick 26.
    
    # Tentative radius for initialization (smaller than expected max to fit easily)
    r_init = 0.09 
    
    centers = []
    
    # Generate points in a hexagonal pattern
    # Rows are shifted horizontally.
    # Vertical spacing = r * sqrt(3)
    # Horizontal spacing = 2 * r
    
    row_height = r_init * np.sqrt(3)
    x_step = 2 * r_init
    
    # We'll generate rows until we have enough points
    y = r_init
    while len(centers) < 30: # Generate slightly more than needed
        # Determine x coordinates for this row
        # Even rows (index 0, 2...) start at r_init
        # Odd rows (index 1, 3...) start at 2*r_init (shifted by r_init)
        row_idx = int((y - r_init) / row_height) if row_height > 0 else 0
        # Actually just track y loop
        # Let's just iterate y
        
        # Check if row fits in height (center must be >= r and <= 1-r)
        if y + r_init > 1 + 1e-6:
            break
            
        # Determine x start offset
        # To fit more circles, odd rows (shifted) can sometimes fit an extra one or just fit tighter?
        # Actually, for a fixed width, shifting allows tighter packing vertically, 
        # but horizontally the constraint is similar.
        # Let's just place points.
        
        # Offset for staggered rows
        offset = 0.0
        # If row index is odd, shift by r_init
        # We need to know row index. 
        # Approx row index:
        current_row_idx = int(round((y - r_init) / row_height))
        
        if current_row_idx % 2 != 0:
            offset = r_init
            
        x = r_init + offset
        while x + r_init <= 1 + 1e-6:
            centers.append([x, y])
            x += x_step
            if len(centers) >= 30:
                break
        y += row_height
        
    # Select 26 centers. 
    # If we have more, pick the first 26. 
    # If fewer (unlikely with r=0.09), we might need to adjust, but 0.09 is small enough.
    centers = centers[:n]
    
    # Convert to numpy array
    centers = np.array(centers)
    radii = np.full(n, r_init)
    
    # Flatten variables for optimizer: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i + 1] = centers[i, 1]
        x0[3*i + 2] = radii[i]

    # ---------------------------------------------------------
    # 2. Optimization Setup
    # ---------------------------------------------------------
    
    # Bounds: x, y in [0, 1], r >= 0. 
    # Also r <= 0.5 implicitly, but x,y bounds handle it.
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    # Constraints
    constraints = []
    
    # Boundary constraints:
    # x >= r  => x - r >= 0
    # x <= 1 - r => x + r - 1 <= 0 => 1 - x - r >= 0
    # Same for y
    for i in range(n):
        # x >= r
        def make_x_ge_r(i):
            def con(x):
                return x[3*i] - x[3*i + 2]
            return con
        constraints.append({'type': 'ineq', 'fun': make_x_ge_r(i)})
        
        # x <= 1 - r  => 1 - x - r >= 0
        def make_x_le_1_minus_r(i):
            def con(x):
                return 1.0 - x[3*i] - x[3*i + 2]
            return con
        constraints.append({'type': 'ineq', 'fun': make_x_le_1_minus_r(i)})

        # y >= r
        def make_y_ge_r(i):
            def con(x):
                return x[3*i + 1] - x[3*i + 2]
            return con
        constraints.append({'type': 'ineq', 'fun': make_y_ge_r(i)})

        # y <= 1 - r
        def make_y_le_1_minus_r(i):
            def con(x):
                return 1.0 - x[3*i + 1] - x[3*i + 2]
            return con
        constraints.append({'type': 'ineq', 'fun': make_y_le_1_minus_r(i)})

    # Overlap constraints: dist >= r_i + r_j
    # dist^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            def make_overlap_con(i, j):
                def con(x):
                    dx = x[3*i] - x[3*j]
                    dy = x[3*i + 1] - x[3*j + 1]
                    ri = x[3*i + 2]
                    rj = x[3*j + 2]
                    return (dx**2 + dy**2) - (ri + rj)**2
                return con
            constraints.append({'type': 'ineq', 'fun': make_overlap_con(i, j)})

    # Objective: Maximize sum(r) => Minimize -sum(r)
    def objective(x):
        s = 0.0
        for i in range(n):
            s += x[3*i + 2]
        return -s

    # ---------------------------------------------------------
    # 3. Run Optimization
    # ---------------------------------------------------------
    
    # SLSQP is a good choice for constrained optimization
    # We might need to run it a couple of times or with options to ensure convergence
    
    # First run
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})
    
    # If not successful or to improve, we can try a second run from the result
    # or just accept. SLSQP is usually robust if initialized well.
    
    best_x = res.x
    best_val = -res.fun
    
    # Optional: Refine with a second run if first failed or to polish
    # Sometimes SLSQP gets stuck.
    if not res.success:
        # Try a simple random perturbation or restart? 
        # Or just try again with slightly different tolerance.
        pass 
    else:
        # We can try to perturb slightly and re-optimize to escape local minima if needed,
        # but for packing, the geometry is usually guiding.
        pass

    # Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = best_x[3*i]
        final_centers[i, 1] = best_x[3*i + 1]
        final_radii[i] = best_x[3*i + 2]
        
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
