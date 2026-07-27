# sol_000035 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8e46300b) state=9d8a35e2 sum of radii=2.583882 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    rng = np.random.default_rng(42) # Fixed seed for reproducibility

    # --- 1. Generate Initial Guess ---
    # A hexagonal lattice is denser than a square grid.
    # We try to fit points in a hexagonal pattern and then optimize.
    # Approximate radius for 26 circles is around 0.1.
    # Let's create a grid of points and select 26, or just generate random points.
    # A structured initialization is usually better.
    
    # Attempt a hexagonal packing layout
    # Rows with alternating counts
    # 6 rows: 5, 4, 5, 4, 5, 4 -> 27 circles. We can drop one.
    # Or 5 rows: 6, 5, 6, 5, 6 -> 28 circles.
    
    # Let's generate a hexagonal grid and pick 26 points.
    r_init = 0.08 # Initial radius guess
    points = []
    
    # Hexagonal packing parameters
    # Horizontal spacing 2*r, Vertical spacing sqrt(3)*r
    # But we are placing centers.
    # Let's just use a dense random initialization perturbed around a grid
    # or a specific pattern.
    
    # Simple approach: Random points in [0,1]x[0,1]
    # But to help convergence, let's place them on a grid first
    # and then the optimizer will move them.
    
    # 6x6 grid has 36 points. We can take a subset.
    # Or just 5x5 grid (25 points) plus one random point.
    # Let's try a 6x6 grid subset.
    
    x_coords = np.linspace(0.1, 0.9, 10)
    y_coords = np.linspace(0.1, 0.9, 10)
    
    # Generate all grid points
    grid_points = np.array([(x, y) for x in x_coords for y in y_coords])
    
    # Select 26 points that are well spread
    # Simplest: take first 26
    init_centers = grid_points[:n]
    
    # Add some random noise to break symmetry
    init_centers += rng.normal(0, 0.02, size=init_centers.shape)
    
    # Clip to valid range for centers (radius 0.08)
    init_centers = np.clip(init_centers, 0.08, 0.92)
    
    init_radii = np.full(n, 0.05) # Start small to ensure feasibility
    
    # --- 2. Define Objective and Constraints ---
    
    def objective(params):
        # params: [x1, y1, r1, x2, y2, r2, ...]
        # We want to maximize sum of radii, so minimize negative sum
        radii = params[2::3]
        return -np.sum(radii)

    def boundary_constraints(params):
        constraints = []
        for i in range(n):
            x = params[3*i]
            y = params[3*i + 1]
            r = params[3*i + 2]
            
            # x - r >= 0  => r - x <= 0 is wrong. x - r >= 0
            # scipy expects constraint >= 0
            constraints.append(x - r - 1e-7)
            constraints.append(1 - x - r - 1e-7)
            constraints.append(y - r - 1e-7)
            constraints.append(1 - y - r - 1e-7)
        return np.array(constraints)

    def overlap_constraints(params):
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi = params[3*i], params[3*i + 1]
                xj, yj = params[3*j], params[3*j + 1]
                ri, rj = params[3*i + 2], params[3*j + 2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                
                # dist >= sum_r  => dist^2 >= sum_r^2
                # dist_sq - sum_r^2 >= 0
                # To avoid square root in gradient, we can use squared distance
                # But constraint is non-smooth at dist=0 if not careful? 
                # Here dist won't be 0 if initialized well.
                constraints.append(dist_sq - (sum_r)**2 - 1e-10) 
        return np.array(constraints)

    # Combine constraints
    # Note: SLSQP can handle arrays of constraints
    # But defining a single function returning array is better
    
    def all_constraints(params):
        c1 = boundary_constraints(params)
        c2 = overlap_constraints(params)
        return np.concatenate([c1, c2])

    # Bounds
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)]) # x, y, r

    # Initial parameters vector
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = init_centers[i, 0]
        x0[3*i + 1] = init_centers[i, 1]
        x0[3*i + 2] = init_radii[i]

    # --- 3. Optimization ---
    
    # Options for SLSQP
    options = {
        'maxiter': 500,
        'ftol': 1e-9,
        'disp': False
    }

    # Run optimization
    # Using SLSQP
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': all_constraints},
        options=options
    )

    # Extract solution
    best_params = res.x
    centers = np.array([[best_params[3*i], best_params[3*i + 1]] for i in range(n)])
    radii = np.array([best_params[3*i + 2] for i in range(n)])

    # --- 4. Post-processing and Validation ---
    
    # Ensure non-negative radii (optimizer might push slightly negative if bound handling is loose)
    radii = np.maximum(radii, 1e-9)
    
    # Re-check validity and fix if necessary
    # If invalid, we might need to scale down slightly
    valid = validate_packing(centers, radii)
    
    if not valid:
        # Fallback: scale down radii until valid
        # This is a brute force fix
        scale = 1.0
        while not valid and scale > 0.1:
            scale *= 0.95
            radii_scaled = radii * scale
            valid = validate_packing(centers, radii_scaled)
        radii = radii_scaled
    
    sum_radii = np.sum(radii)
    
    # Return result
    return centers, radii, sum_radii

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    import numpy as np
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        # print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        # print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            # print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            # print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            # print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                # print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True
