# sol_000009 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9f77b693) state=9481d226 sum of radii=2.340000 correctness=1.0
# stdout(first 200): (array([[0.09      , 0.09      ],        [0.27      , 0.09      ],        [0.45      , 0.09      ],        [0.63      , 0.09      ],        [0.81      , 0.09      ],        [0.18      , 0.24588457],  
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26

    # 1. Initialization: Hexagonal Packing Strategy
    # We aim for a staggered grid to maximize density.
    # A 6-row configuration with counts [5, 5, 5, 5, 4, 2] sums to 26.
    row_counts = [5, 5, 5, 5, 4, 2]
    
    centers = []
    # Initial radius estimate for hexagonal packing in a unit square
    # Height constraint: 2r + 5 * sqrt(3) * r <= 1  => r <= 1 / (2 + 5*sqrt(3)) approx 0.09
    # We start slightly smaller to ensure valid placement before optimization
    initial_r = 0.09 
    
    y_pos = initial_r
    row_height = np.sqrt(3) * initial_r
    
    for i, count in enumerate(row_counts):
        # Determine row shift for staggered effect
        # Even rows (0, 2, 4) start at x = r
        # Odd rows (1, 3, 5) start at x = 2r (shifted by one radius)
        x_start = initial_r if i % 2 == 0 else 2 * initial_r
        
        row_centers = []
        for j in range(count):
            x_pos = x_start + j * (2 * initial_r)
            # If a circle is pushed out of bounds by the stagger, we clamp it or skip logic
            # But with these counts, it should fit.
            # Just in case, ensure x_pos is valid
            if x_pos + initial_r <= 1.0:
                row_centers.append([x_pos, y_pos])
        
        # If we didn't fit enough circles in this row due to boundary, try to fit remaining
        # But the counts are designed to fit.
        # However, the last row has 2 circles. 
        # If row 5 (index 5) is odd, it starts at 2r.
        # 2 circles at 2r and 4r. 4r + r = 5r = 0.45 <= 1. OK.
        
        centers.extend(row_centers)
        y_pos += row_height

    # Ensure we have exactly 26 centers
    # Fallback if generation logic missed any (unlikely with these params)
    while len(centers) < n:
        centers.append([0.5, 0.5])
    centers = np.array(centers[:n])
    radii = np.full(n, initial_r)

    # 2. Optimization Setup
    # Variables: [x1, y1, ..., x26, y26, r1, ..., r26] -> Total 78 variables
    # However, optimizing equal radii is often sufficient and faster.
    # Let's try equal radii first, as the sum of radii is linear with r.
    # Variables: [x1, y1, ..., x26, y26, r] -> Total 53 variables
    
    x0 = np.hstack([centers.flatten(), initial_r])
    
    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)]

    def objective(vars):
        # Maximize sum of radii (minimize negative sum)
        r = vars[-1]
        return -26.0 * r

    def constraint_inside(vars):
        # x_i - r >= 0  => r - x_i <= 0
        # 1 - x_i - r >= 0 => x_i + r - 1 <= 0
        # Same for y
        centers_opt = vars[:2*n].reshape(n, 2)
        r = vars[-1]
        
        violations = []
        for i in range(n):
            x, y = centers_opt[i]
            # x - r >= 0
            violations.append(r - x)
            # x + r <= 1
            violations.append(x + r - 1.0)
            # y - r >= 0
            violations.append(r - y)
            # y + r <= 1
            violations.append(y + r - 1.0)
            
        # SLSQP expects g(x) >= 0. 
        # We transform inequalities to >= 0 form.
        # x >= r => x - r >= 0
        # 1 - x - r >= 0
        res = []
        for i in range(n):
            x, y = centers_opt[i]
            res.append(x - r)
            res.append(1.0 - x - r)
            res.append(y - r)
            res.append(1.0 - y - r)
        return np.array(res)

    def constraint_non_overlap(vars):
        centers_opt = vars[:2*n].reshape(n, 2)
        r = vars[-1]
        dist_sum = 2.0 * r
        
        res = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j])**2))
                # dist >= 2r => dist - 2r >= 0
                res.append(dist - dist_sum)
        return np.array(res)

    # Define constraints for SLSQP
    constraints = []
    
    # Boundary constraints
    # Non-overlap constraints (can be many, ~300)
    # To speed up, we can pass them as a single function returning an array
    
    cons_boundary = {'type': 'ineq', 'fun': constraint_inside}
    cons_overlap = {'type': 'ineq', 'fun': constraint_non_overlap}
    
    constraints.append(cons_boundary)
    constraints.append(cons_overlap)

    # 3. Run Optimizer
    # Method 'SLSQP' is suitable for non-linear constrained optimization
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
    )

    # Extract results
    if res.success or res.fun < -2.5: # Check if we improved significantly
        final_vars = res.x
        final_centers = final_vars[:2*n].reshape(n, 2)
        final_r = final_vars[-1]
        final_radii = np.full(n, final_r)
    else:
        # Fallback to initial if optimization failed
        final_centers = centers
        final_radii = radii

    sum_radii = np.sum(final_radii)

    return final_centers, final_radii, sum_radii

# Execute to return the solution
print(run_packing())
