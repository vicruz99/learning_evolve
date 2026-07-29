# sol_000094 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e1ebaf70) state=1988c397 sum of radii=2.590061 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    # --- Phase 1: Initialization (Hexagonal Lattice) ---
    # We define a hexagonal pattern that fits 26 circles.
    # Rows: 5, 5, 5, 5, 4, 2
    rows = [5, 5, 5, 5, 4, 2]
    centers_init = []
    r_init = 0.05  # Small initial radius to ensure no overlap

    # Hexagonal packing geometry
    spacing_x = 2 * r_init * 1.5  # Horizontal distance between centers in a row (loose)
    spacing_y = 2 * r_init * np.sqrt(3) / 2  # Vertical distance between rows
    shift_x = r_init * 1.5  # Horizontal shift for alternating rows

    x_start, y_start = r_init, r_init
    current_y = y_start

    for i, count in enumerate(rows):
        # Center the row horizontally
        row_width = (count - 1) * spacing_x
        current_x = (1 - row_width) / 2
        
        # Apply shift for alternating rows (1, 3, 5...)
        if i % 2 == 1:
            current_x += spacing_x / 2
            
        for _ in range(count):
            centers_init.append([current_x, current_y])
            current_x += spacing_x
        current_y += spacing_y

    centers_init = np.array(centers_init)

    # --- Phase 2: Optimization ---
    # We define a function to minimize: Negative sum of radii
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26] (78 variables)
    
    def objective(params):
        radii = params[2::3]
        return -np.sum(radii)

    def constraints(params):
        constraints_list = []
        centers = np.array([params[i:i+2] for i in range(0, 78, 3)])
        radii = params[2::3]
        
        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        for i in range(n):
            constraints_list.append(centers[i, 0] - radii[i]) # x >= r
            constraints_list.append(1 - centers[i, 0] - radii[i]) # 1-x >= r
            constraints_list.append(centers[i, 1] - radii[i]) # y >= r
            constraints_list.append(1 - centers[i, 1] - radii[i]) # 1-y >= r
        
        # Non-overlap constraints: dist >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.sum((centers[i] - centers[j])**2)
                r_sum_sq = (radii[i] + radii[j])**2
                constraints_list.append(dist_sq - r_sum_sq)
        
        return np.array(constraints_list)

    # Initial parameters vector
    x0 = np.zeros(78)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = r_init

    # Bounds: x, y in [0, 1], r >= 0
    bounds = [(0, 1)] * (2 * n) + [(0, 1)] * n

    # Use SLSQP which handles constraints well
    # We set a high number of max iterations to allow convergence
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints={'type': 'ineq', 'fun': constraints},
        options={'maxiter': 500, 'disp': False, 'ftol': 1e-12}
    )

    # Extract results
    final_params = result.x
    centers = np.array([[final_params[3*i], final_params[3*i+1]] for i in range(n)])
    radii = np.array([final_params[3*i+2] for i in range(n)])
    sum_radii = np.sum(radii)

    return centers, radii, sum_radii
