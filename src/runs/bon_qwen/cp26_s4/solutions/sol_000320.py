# sol_000320 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f00b2e18) state=f6f94bd3 sum of radii=2.607988 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    """
    n_circles = 26
    dim = 2
    total_vars = n_circles * (dim + 1) # x, y, r for each circle
    
    # Initial guess
    # We start with a hexagonal-like packing with radius slightly larger than 0.1
    # to encourage the optimizer to find a tight packing.
    # We will generate points and then let the optimizer adjust.
    
    # Generate a hexagonal grid initialization
    # Target radius for initialization
    r_init = 0.105
    
    centers_init = []
    radii_init = []
    
    # Hexagonal packing logic
    # Vertical spacing
    dy = np.sqrt(3) * r_init
    # Horizontal spacing
    dx = 2 * r_init
    
    # Try to fill rows
    row_y = r_init
    circle_count = 0
    
    while circle_count < n_circles:
        # Determine number of circles in this row
        # Alternating shift
        if len(centers_init) > 0:
            # Check previous row's x positions to determine shift
            # If previous row started at r_init, this one starts at 2*r_init (shifted)
            # Actually, standard hex: row 0 at x=r, row 1 at x=2r, row 2 at x=r...
            # Let's just alternate
            prev_len = len(centers_init)
            # Estimate row index
            row_idx = prev_len // 5 # approximate
            if row_idx % 2 == 1:
                x_start = 2 * r_init
            else:
                x_start = r_init
        else:
            x_start = r_init
            
        # Max x allowed
        x_current = x_start
        row_circles = 0
        while x_current + r_init <= 1.0 and circle_count < n_circles:
            centers_init.append([x_current, row_y])
            radii_init.append(r_init)
            circle_count += 1
            x_current += dx
            row_circles += 1
            
        row_y += dy

    # If we didn't reach 26, or overshot (though loop condition handles it), adjust
    # If we have fewer than 26, add random ones or fill gaps
    # With r=0.105, 5 rows * 5 cols might not fit perfectly due to width.
    # Let's ensure we have exactly 26 by padding if necessary
    while len(centers_init) < n_circles:
        # Add a circle in the center or a gap
        centers_init.append([0.5, 0.5])
        radii_init.append(r_init)
        
    # Trim if more (shouldn't happen with logic above but safety)
    centers_init = centers_init[:n_circles]
    radii_init = radii_init[:n_circles]
    
    # Reshape to vector for scipy
    # Format: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(total_vars)
    for i in range(n_circles):
        x0[3*i] = centers_init[i][0]
        x0[3*i+1] = centers_init[i][1]
        x0[3*i+2] = radii_init[i]
        
    # Bounds
    # x, y in [0, 1], r >= 0
    # Ideally x in [r, 1-r], but that's a constraint. 
    # Bounds for solver:
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 1.0)) # r (upper bound 1 is safe)

    # Constraints
    constraints = []
    
    # 1. Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    for i in range(n_circles):
        # x - r >= 0 => x - r
        cons_x_min = {
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i+2]
        }
        constraints.append(cons_x_min)
        
        # x + r <= 1 => 1 - (x + r)
        cons_x_max = {
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - (v[3*i] + v[3*i+2])
        }
        constraints.append(cons_x_max)
        
        # y - r >= 0
        cons_y_min = {
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]
        }
        constraints.append(cons_y_min)
        
        # y + r <= 1
        cons_y_max = {
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - (v[3*i+1] + v[3*i+2])
        }
        constraints.append(cons_y_max)

    # 2. Non-overlap constraints: dist >= r_i + r_j
    # dist^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            cons_overlap = {
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 
                    - (v[3*i+2] + v[3*j+2])**2
                )
            }
            constraints.append(cons_overlap)

    # Objective: Maximize sum of radii => Minimize -sum(radii)
    def objective(v):
        total_r = 0.0
        for i in range(n_circles):
            total_r += v[3*i+2]
        return -total_r

    # Optimization
    # Using SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                   options={'maxiter': 1000, 'ftol': 1e-9})
    
    if not res.success:
        print("Optimization warning:", res.message)

    # Extract results
    best_centers = np.zeros((n_circles, 2))
    best_radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        best_centers[i][0] = res.x[3*i]
        best_centers[i][1] = res.x[3*i+1]
        best_radii[i] = res.x[3*i+2]
        
    # Clip tiny negative radii if any (shouldn't happen due to bounds)
    best_radii = np.maximum(best_radii, 0.0)
    
    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, sum_radii

if __name__ == "__main__":
    centers, radii, s_r = run_packing()
    print(f"Sum of radii: {s_r}")
    
    # Local validation
    import math
    def validate_packing_local(centers, radii):
        n = centers.shape[0]
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
                if dist < radii[i] + radii[j] - 1e-9:
                    return False
        return True

    if validate_packing_local(centers, radii):
        print("Valid packing.")
    else:
        print("Invalid packing.")
