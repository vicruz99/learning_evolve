# sol_000009 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 92133c71) state=c626080d sum of radii=2.460107 correctness=1.0
# stdout(first 200): Optimization terminated successfully    (Exit mode 0)             Current function value: -2.460106896902753             Iterations: 21             Function evaluations: 1582             Gradient eval
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal Lattice
    # We want to fill the square with a hexagonal pattern.
    # We start with a small radius to ensure feasibility, then let optimizer grow them.
    # A safe initial radius is 0.05.
    r_init = 0.05
    
    # We will generate points in rows.
    # Hexagonal spacing: vertical distance between rows is r * sqrt(3).
    # Horizontal distance between circles in a row is 2 * r.
    # Rows are shifted by r horizontally.
    
    # Let's determine how many rows we can fit with a slightly larger radius to guide placement,
    # but we'll use r_init for actual coordinates to be safe.
    # Actually, let's just generate a dense grid pattern that fits.
    # A 5x5 grid plus some extra points is a good start, or a strict hex grid.
    
    centers_list = []
    
    # Try to fit points in a hex grid.
    # Row height step
    dy = r_init * np.sqrt(3)
    # Col width step
    dx = 2 * r_init
    
    # Start y at r_init
    y = r_init
    row_idx = 0
    points_generated = 0
    
    while points_generated < n_circles:
        # Determine x offset for this row
        # Even rows (0, 2, ...) start at r_init (left aligned with margin)
        # Odd rows (1, 3, ...) start at 2*r_init (shifted by r_init)
        # Actually standard hex: shift is r. 
        # If row 0 starts at r, row 1 starts at 2r? 
        # Distance between (r, r) and (2r, r + r*sqrt(3)) is sqrt(r^2 + 3r^2) = 2r. Correct.
        
        x_offset = r_init if row_idx % 2 == 0 else 2 * r_init
        
        x = x_offset
        while x <= 1 - r_init:
            centers_list.append([x, y])
            points_generated += 1
            if points_generated >= n_circles:
                break
            x += dx
        
        y += dy
        row_idx += 1
        
        # Safety break to avoid infinite loop if logic fails
        if y > 1.0 + r_init:
            break
            
    # If we didn't generate enough points (unlikely with small r), fill rest randomly or grid
    # But with r=0.05, we will definitely generate > 26.
    
    # Trim or pad to exactly 26
    centers_list = centers_list[:n_circles]
    
    # Convert to numpy array
    centers = np.array(centers_list)
    radii = np.full(n_circles, r_init)
    
    # Flatten for optimizer: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # 2. Define Objective
    def objective(vars):
        # Maximize sum of radii -> Minimize negative sum
        radii_vals = vars[2::3]
        return -np.sum(radii_vals)
        
    # 3. Define Constraints
    # SLSQP takes constraints as dictionaries: {'type': 'ineq', 'fun': fun}
    # Inequality constraints must be >= 0.
    
    constraints = []
    
    # Boundary constraints:
    # x >= r  => x - r >= 0
    # x <= 1 - r => x + r - 1 <= 0 => 1 - x - r >= 0
    # Same for y
    
    def add_boundary_constraints():
        cons = []
        for i in range(n_circles):
            idx_x = 3*i
            idx_y = 3*i + 1
            idx_r = 3*i + 2
            
            # x - r >= 0
            def c1(v, i=i): return v[3*i] - v[3*i+2]
            cons.append({'type': 'ineq', 'fun': c1})
            
            # 1 - x - r >= 0
            def c2(v, i=i): return 1.0 - v[3*i] - v[3*i+2]
            cons.append({'type': 'ineq', 'fun': c2})
            
            # y - r >= 0
            def c3(v, i=i): return v[3*i+1] - v[3*i+2]
            cons.append({'type': 'ineq', 'fun': c3})
            
            # 1 - y - r >= 0
            def c4(v, i=i): return 1.0 - v[3*i+1] - v[3*i+2]
            cons.append({'type': 'ineq', 'fun': c4})
        return cons

    constraints.extend(add_boundary_constraints())
    
    # Pairwise non-overlap constraints:
    # dist^2 >= (r1 + r2)^2
    # (x1-x2)^2 + (y1-y2)^2 - (r1+r2)^2 >= 0
    
    def add_pairwise_constraints():
        cons = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                def c_pair(v, i=i, j=j):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    r_sum_sq = (ri + rj)**2
                    return dist_sq - r_sum_sq
                cons.append({'type': 'ineq', 'fun': c_pair})
        return cons

    constraints.extend(add_pairwise_constraints())

    # 4. Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius)
    bounds = []
    for i in range(n_circles):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # 5. Run Optimization
    # SLSQP is good for smooth nonlinear constraints
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'disp': True})
    
    # 6. Extract results
    final_vars = result.x
    centers_final = np.zeros((n_circles, 2))
    radii_final = np.zeros(n_circles)
    
    for i in range(n_circles):
        centers_final[i, 0] = final_vars[3*i]
        centers_final[i, 1] = final_vars[3*i+1]
        radii_final[i] = final_vars[3*i+2]
        
    sum_radii = np.sum(radii_final)
    
    # Ensure validity check (optional but good for debugging)
    # The validate_packing function will be called externally, but we can trust the optimizer
    # provided constraints were respected.
    
    return centers_final, radii_final, sum_radii

# Run the packing to verify
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(centers)
    # print(radii)
