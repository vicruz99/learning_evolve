# sol_000086 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9821b492) state=3a6500a4 sum of radii=2.596595 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses scipy.optimize to solve the nonlinear programming problem.
    """
    n = 26
    
    # 1. Initialize centers in a hexagonal-like grid to provide a good starting point
    # This helps the optimizer converge to a better local optimum.
    # We aim for a layout that fits roughly in the square.
    # A pattern of rows with counts 5, 4, 5, 4, 5, 3 sums to 26.
    
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.01 # Start with small radii
    
    # Define row structure
    # Rows: 5, 4, 5, 4, 5, 3
    row_counts = [5, 4, 5, 4, 5, 3]
    
    # Estimate spacing for initialization
    # We want to fit in [0,1]x[0,1].
    # Let's place them centered.
    
    # Vertical spacing
    # 6 rows. Let's distribute y from 0.1 to 0.9
    y_coords = np.linspace(0.1, 0.9, len(row_counts))
    
    idx = 0
    for i, count in enumerate(row_counts):
        y = y_coords[i]
        # Horizontal spacing
        # Shift odd rows (1, 3, 5) to create hexagonal packing
        if i % 2 == 1:
            # Shifted row
            # x range for count circles. 
            # Let's center them.
            # Width available approx 0.8. 
            # Spacing dx = 0.8 / (count + 1) ?
            # Let's just distribute evenly in [0.1, 0.9]
            x_start = 0.1 + 0.1 # Shift
            x_end = 0.9 - 0.1
            if count > 1:
                x_coords = np.linspace(x_start, x_end, count)
            else:
                x_coords = np.array([0.5])
        else:
            # Aligned row
            x_start = 0.1
            x_end = 0.9
            if count > 1:
                x_coords = np.linspace(x_start, x_end, count)
            else:
                x_coords = np.array([0.5])
        
        for x in x_coords:
            centers[idx, 0] = x
            centers[idx, 1] = y
            idx += 1
            
    # Initial radii guess. 
    # Based on grid, r ~ 0.05. 
    # We will let optimizer find better.
    initial_r = 0.05
    for i in range(n):
        radii[i] = initial_r
        
    # 2. Prepare optimization variables
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # Shape: (n * 3,)
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # 3. Define Objective Function
    # Maximize sum of radii => Minimize -sum(radii)
    def objective(vars):
        r_sum = 0
        for i in range(n):
            r_sum += vars[3*i + 2]
        return -r_sum

    # 4. Define Constraints
    # Constraints must be >= 0
    
    # Boundary constraints:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    
    # Pairwise non-overlap constraints:
    # dist(i, j) - (r_i + r_j) >= 0
    # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    # To avoid sqrt in constraints (smoothness), we can square it?
    # (xi-xj)^2 + (yi-yj)^2 >= (ri + rj)^2
    # (xi-xj)^2 + (yi-yj)^2 - (ri + rj)^2 >= 0
    
    # However, squaring can introduce numerical issues or local minima if not careful.
    # But for valid packings where dist > 0, it's equivalent.
    # Let's use the squared distance constraint.
    
    constraints = []
    
    # Add boundary constraints
    for i in range(n):
        xi_idx = 3*i
        yi_idx = 3*i + 1
        ri_idx = 3*i + 2
        
        # x - r >= 0
        def make_boundary_x_ge(i):
            return lambda vars: vars[3*i] - vars[3*i + 2]
        
        # 1 - x - r >= 0
        def make_boundary_x_le(i):
            return lambda vars: 1.0 - vars[3*i] - vars[3*i + 2]
            
        # y - r >= 0
        def make_boundary_y_ge(i):
            return lambda vars: vars[3*i + 1] - vars[3*i + 2]
            
        # 1 - y - r >= 0
        def make_boundary_y_le(i):
            return lambda vars: 1.0 - vars[3*i + 1] - vars[3*i + 2]
        
        constraints.append({'type': 'ineq', 'fun': make_boundary_x_ge(i)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_x_le(i)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_y_ge(i)})
        constraints.append({'type': 'ineq', 'fun': make_boundary_y_le(i)})

    # Add pairwise constraints
    # Using squared distance to avoid sqrt in derivative if possible, 
    # though scipy handles sqrt fine. Squared is polynomial, nicer.
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, ri = 3*i, 3*i + 1, 3*i + 2
            xj, yj, rj = 3*j, 3*j + 1, 3*j + 2
            
            def make_pairwise_constraint(i, j):
                # Returns (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2
                def fun(vars):
                    dx = vars[3*i] - vars[3*j]
                    dy = vars[3*i + 1] - vars[3*j + 1]
                    dr = vars[3*i + 2] + vars[3*j + 2]
                    return dx*dx + dy*dy - dr*dr
                return fun
            
            constraints.append({'type': 'ineq', 'fun': make_pairwise_constraint(i, j)})

    # 5. Bounds for variables
    # x, y in [0, 1], r >= 0
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 1)) # r (radius can't be > 0.5 really, but 1 is safe)
        
    # 6. Run Optimization
    # Method SLSQP is good for constrained nonlinear optimization
    # maxiter might need to be high.
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})
    
    # 7. Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = res.x[3*i]
        final_centers[i, 1] = res.x[3*i + 1]
        final_radii[i] = res.x[3*i + 2]
        
    sum_radii = np.sum(final_radii)
    
    # Validate internally to be sure (optional but good practice)
    # If result is invalid due to optimization failure, we might return initial?
    # But SLSQP should respect constraints.
    
    return final_centers, final_radii, sum_radii
