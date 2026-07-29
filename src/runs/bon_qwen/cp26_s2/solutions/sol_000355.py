# sol_000355 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4c8413f9) state=0c70b4fe sum of radii=2.608631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal Packing
    # We create a staggered grid pattern. 
    # Pattern of counts per row: 5, 4, 5, 4, 5, 3 sums to 26.
    # This fits well in a square if radius is small enough.
    
    rows_counts = [5, 4, 5, 4, 5, 3]
    centers = []
    
    # Initial radius guess. 0.09 is safe for this configuration.
    # Target is around 0.101, so we start slightly lower.
    r_init = 0.09
    
    y_pos = r_init
    sqrt3 = np.sqrt(3)
    
    for idx, count in enumerate(rows_counts):
        # Determine x-start for this row
        # Even rows (0, 2, 4) have 5 circles, start at r
        # Odd rows (1, 3) have 4 circles, shifted by r (start at 2r? No, centers at 2r, 4r...)
        # Actually, standard hex packing:
        # Row 0: x = r, 3r, 5r, 7r, 9r
        # Row 1: x = 2r, 4r, 6r, 8r (shifted by r relative to neighbors)
        
        if idx % 2 == 0:
            # Full row (or close to)
            # Start x = r
            x_start = r_init
            step = 2 * r_init
        else:
            # Staggered row
            # Start x = 2r (shifted by r)
            x_start = 2 * r_init
            step = 2 * r_init
            
        for i in range(count):
            x_pos = x_start + i * step
            centers.append([x_pos, y_pos])
            
        # Move to next row
        y_pos += sqrt3 * r_init
        
    centers = np.array(centers)
    radii = np.full(n_circles, r_init)
    
    # 2. Optimization
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # Total variables = 3 * 26 = 78
    
    def objective(vars):
        # We want to maximize sum of radii, so minimize negative sum
        r_vars = vars[2::3]
        return -np.sum(r_vars)
    
    def constraint_overlap(vars):
        # dist >= r_i + r_j  =>  dist - (r_i + r_j) >= 0
        # We return a vector of values that must be >= 0
        # However, for SLSQP, 'ineq' constraints must be >= 0.
        # There are n*(n-1)/2 pairs. 26*25/2 = 325 constraints.
        # This might be heavy but manageable.
        
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        constraints = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                val = dist - (r[i] + r[j])
                constraints.append(val)
        return np.array(constraints)

    def constraint_boundary_x(vars):
        # r <= x <= 1-r
        # x - r >= 0
        # 1 - x - r >= 0
        x = vars[0::3]
        r = vars[2::3]
        return np.concatenate([x - r, 1 - x - r])

    def constraint_boundary_y(vars):
        y = vars[1::3]
        r = vars[2::3]
        return np.concatenate([y - r, 1 - y - r])

    def constraint_radius(vars):
        # r >= 0
        r = vars[2::3]
        return r

    # Initial variables
    x0 = np.zeros(3 * n_circles)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii
    
    # Define constraints for SLSQP
    cons = [
        {'type': 'ineq', 'fun': constraint_overlap},
        {'type': 'ineq', 'fun': constraint_boundary_x},
        {'type': 'ineq', 'fun': constraint_boundary_y},
        {'type': 'ineq', 'fun': constraint_radius}
    ]
    
    # Bounds for variables to help solver (optional but good)
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n_circles) + [(0, 0.5) * n_circles]
    # Actually bounds is list of tuples.
    bounds = []
    for i in range(n_circles):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r

    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                          options={'maxiter': 1000, 'ftol': 1e-9})
        
        if result.success:
            x_opt = result.x[0::3]
            y_opt = result.x[1::3]
            r_opt = result.x[2::3]
        else:
            # Fallback to initial if optimization fails, though unlikely
            x_opt = centers[:, 0]
            y_opt = centers[:, 1]
            r_opt = radii
            
    except Exception:
        # Fallback
        x_opt = centers[:, 0]
        y_opt = centers[:, 1]
        r_opt = radii

    # 3. Post-processing / Validation adjustment
    # Ensure strict validity within tolerance
    # Clamp radii to be non-negative
    r_opt = np.maximum(r_opt, 0)
    
    # Adjust positions to be inside bounds strictly
    for i in range(n_circles):
        r = r_opt[i]
        x_opt[i] = np.clip(x_opt[i], r, 1 - r)
        y_opt[i] = np.clip(y_opt[i], r, 1 - r)
        
        # If r is too large for the position, shrink r (should not happen if constraints held)
        if x_opt[i] < r or x_opt[i] > 1 - r:
             r_opt[i] = min(x_opt[i], 1 - x_opt[i])
        if y_opt[i] < r or y_opt[i] > 1 - y_opt[i]:
             r_opt[i] = min(r_opt[i], min(y_opt[i], 1 - y_opt[i]))

    # Final check for overlaps and slight corrections
    # If small overlaps persist due to numerical issues, shrink radii slightly
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt((x_opt[i] - x_opt[j])**2 + (y_opt[i] - y_opt[j])**2)
            min_dist = r_opt[i] + r_opt[j]
            if dist < min_dist - 1e-10:
                # Overlap detected, reduce radii
                # Distribute reduction
                reduction = (min_dist - dist) / 2
                r_opt[i] -= reduction
                r_opt[j] -= reduction
                r_opt[i] = max(r_opt[i], 0)
                r_opt[j] = max(r_opt[j], 0)

    centers_opt = np.column_stack((x_opt, y_opt))
    sum_radii = np.sum(r_opt)
    
    return centers_opt, r_opt, sum_radii

# Helper to run and print
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Verify
    # import numpy as np
    # print(validate_packing(c, r))
