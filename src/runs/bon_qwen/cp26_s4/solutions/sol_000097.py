# sol_000097 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e1ebaf70) state=94e2aef0 sum of radii=2.597699 correctness=1.0
# stdout(first 200): Optimization terminated successfully    (Exit mode 0)             Current function value: -2.59769938730869             Iterations: 14             Function evaluations: 1037             Gradient evalu
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26

    # 1. Generate initial hexagonal lattice configuration
    # Pattern: 6, 5, 6, 5, 4 circles in rows
    row_counts = [6, 5, 6, 5, 4]
    centers = []
    
    # Initial spacing parameters for hexagonal packing
    # We start with a slightly overlapping configuration that the optimizer will fix.
    # A radius of roughly 0.1 is our target baseline.
    r_init = 0.1
    dx = 2 * r_init
    dy = np.sqrt(3) * r_init
    
    # Vertical position of the first row center
    y_start = r_init
    
    for k, count in enumerate(row_counts):
        y = y_start + k * dy
        
        # Horizontal shift for hexagonal staggering
        # Even rows (0, 2, 4) start at r_init
        # Odd rows (1, 3) shift by r_init
        if k % 2 == 0:
            x_start = r_init
        else:
            x_start = 2 * r_init # Shifted by one radius relative to even rows
            
        # Adjust x_start for rows with fewer circles to center them
        # This helps in utilizing space better
        if count < 6:
            # Shift left to center the row
            # Max width for 6 circles is approx 1.0
            # Width for 'count' circles is (count-1)*2r + 2r = 2*count*r
            # We want this centered in [0, 1]
            # Center of row should be 0.5
            row_width = (count - 1) * dx + 2 * r_init
            x_start = (1 - row_width) / 2 + r_init
        
        for i in range(count):
            x = x_start + i * dx
            centers.append([x, y])
            
    centers = np.array(centers)
    
    # 2. Prepare for optimization
    # Variables vector: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    x0 = []
    for i in range(n_circles):
        x0.extend([centers[i, 0], centers[i, 1], r_init])
    x0 = np.array(x0)
    
    # Bounds for variables: x, y in [0, 1], r >= 0
    bounds = []
    for _ in range(n_circles):
        bounds.extend([
            (0, 1), # x
            (0, 1), # y
            (0, 1)  # r (upper bound 1 is loose but safe)
        ])
        
    # 3. Define constraints
    # We use scipy.optimize.LinearConstraint or NonlinearConstraint.
    # Since we have many non-linear constraints (circle-circle distance),
    # we define them explicitly.
    
    constraints = []
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    # These can be handled via bounds, but explicit non-linear constraints ensure robustness
    # Actually, bounds (0,1) for x,y and r>=0 are not enough.
    # We need x >= r, x <= 1-r => x - r >= 0 and 1 - x - r >= 0.
    # Let's add these as non-linear constraints.
    
    for i in range(n_circles):
        idx = i * 3
        # x - r >= 0
        def fun_left(i=i):
            return lambda v: v[3*i] - v[3*i + 2]
        constraints.append({'type': 'ineq', 'fun': fun_left()})
        
        # 1 - x - r >= 0
        def fun_right(i=i):
            return lambda v: 1 - v[3*i] - v[3*i + 2]
        constraints.append({'type': 'ineq', 'fun': fun_right()})
        
        # y - r >= 0
        def fun_bottom(i=i):
            return lambda v: v[3*i + 1] - v[3*i + 2]
        constraints.append({'type': 'ineq', 'fun': fun_bottom()})
        
        # 1 - y - r >= 0
        def fun_top(i=i):
            return lambda v: 1 - v[3*i + 1] - v[3*i + 2]
        constraints.append({'type': 'ineq', 'fun': fun_top()})

    # Circle-circle non-overlap constraints
    # dist(i, j) >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    # To avoid sqrt in constraints (better for gradients), we can use squared distance,
    # but SLSQP handles standard forms better. 
    # Using squared: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    # This is equivalent to (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            def fun_overlap(i=i, j=j):
                return lambda v: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            constraints.append({'type': 'ineq', 'fun': fun_overlap()})

    # 4. Objective function
    # Maximize sum of radii => Minimize negative sum
    def objective(v):
        radii = v[2::3]
        return -np.sum(radii)

    # 5. Run Optimization
    # SLSQP is suitable for this type of problem
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'disp': True})
    
    # 6. Extract results
    if result.success or (result.fun < 0): # Check if we got a valid packing (negative objective means positive sum)
        final_centers = np.zeros((n_circles, 2))
        final_radii = np.zeros(n_circles)
        
        for i in range(n_circles):
            final_centers[i, 0] = result.x[3*i]
            final_centers[i, 1] = result.x[3*i+1]
            final_radii[i] = result.x[3*i+2]
            
        sum_radii = np.sum(final_radii)
        return final_centers, final_radii, sum_radii
    else:
        # Fallback to initial configuration if optimization fails
        # Adjust radii to be valid (small enough)
        valid_radii = np.full(n_circles, 0.01)
        return centers, valid_radii, np.sum(valid_radii)
