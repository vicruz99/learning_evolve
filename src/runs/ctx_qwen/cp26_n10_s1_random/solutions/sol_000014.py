# sol_000014 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f64c520b) state=27039d1b sum of radii=2.550069 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    np.random.seed(42)

    # --- Step 1: Generate Initial Hexagonal Layout ---
    # We generate a hexagonal lattice pattern and place 26 circles in it.
    # Then we scale/translate to fit within the unit square boundaries safely.
    centers = []
    r_init = 0.08 # Initial small radius to ensure valid placement
    
    # Generate points in a hexagonal grid
    # Spacing based on initial radius
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    
    # Generate a sufficient number of grid points
    points = []
    y = r_init
    while y < 1.0:
        x = r_init
        row_offset = (int((y - r_init) / dy) % 2) * r_init
        while x < 1.0:
            points.append((x + row_offset, y))
            x += 2 * r_init
        y += dy
        
    # Take the first 26 points and ensure they are inside [0,1]
    # We can shift/scale if necessary, but with r_init=0.08 they should fit well.
    # Just clip to be safe and sort for determinism
    points = np.array(points[:n])
    
    # If we didn't get enough points (unlikely with r=0.08), fallback to random
    if len(points) < n:
        points = np.random.uniform(0, 1, size=(n, 2))

    # Initialize radii
    radii = np.full(n, r_init)

    # --- Step 2: Optimization ---
    # Variables: [x0, y0, r0, x1, y1, r1, ...]
    # Total 3 * n variables
    
    def objective(vars):
        # vars is 1D array of length 3*n
        # We want to maximize sum(r), so minimize -sum(r)
        rs = vars[2::3]
        return -np.sum(rs)

    def constraint_boundary(vars):
        # Constraints: r <= x <= 1-r, r <= y <= 1-r
        # x - r >= 0  => x - r
        # r - x + 1 >= 0 => 1 - (x + r)
        # Same for y
        
        xs = vars[0::3]
        ys = vars[1::3]
        rs = vars[2::3]
        
        con1 = xs - rs
        con2 = 1.0 - (xs + rs)
        con3 = ys - rs
        con4 = 1.0 - (ys + rs)
        
        # Return all constraints as a single array (SLSQP handles vector constraints)
        return np.concatenate([con1, con2, con3, con4])

    def constraint_no_overlap(vars):
        # Constraint: dist(i, j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        
        xs = vars[0::3]
        ys = vars[1::3]
        rs = vars[2::3]
        
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx**2 + dy**2
                r_sum = rs[i] + rs[j]
                constraints.append(dist_sq - r_sum**2)
        return np.array(constraints)

    # Initial vector
    x0 = np.zeros(3 * n)
    x0[0::3] = points[:, 0]
    x0[1::3] = points[:, 1]
    x0[2::3] = radii

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius in unit square)
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    # Define constraints for SLSQP
    # SLSQP expects constraints as dictionaries
    cons = []
    cons.append({'type': 'ineq', 'fun': constraint_boundary})
    cons.append({'type': 'ineq', 'fun': constraint_no_overlap})

    # Run optimization
    # maxiter can be high to allow convergence
    result = scipy.optimize.minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
    )

    # Extract results
    if result.success or result.fun < 0: # Check if we improved or found something valid
        centers = np.column_stack((result.x[0::3], result.x[1::3]))
        radii = result.x[2::3]
    else:
        # Fallback if optimization fails (rare)
        centers = points
        radii = np.full(n, 0.05)

    sum_radii = np.sum(radii)

    # Validate and potentially fix numerical issues
    # The validate function checks for overlaps with tolerance 1e-12.
    # We can slightly shrink radii if needed, but SLSQP usually stays within bounds.
    
    return centers, radii, sum_radii
