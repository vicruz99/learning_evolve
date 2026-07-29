# sol_000037 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 16d9c155) state=97274692 sum of radii=2.590543 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Initialization: Hexagonal Lattice
    # We generate a hexagonal grid and select 26 points that fit in the square.
    n_circles = 26
    centers = []
    r_initial = 0.12
    
    # Generate points in a hexagonal pattern
    y = r_initial
    while True:
        x = r_initial
        row_count = 0
        while x + r_initial <= 1.0:
            centers.append([x, y])
            x += r_initial * np.sqrt(3)
            row_count += 1
            if len(centers) >= n_circles:
                break
        y += r_initial * 1.5
        if len(centers) >= n_circles:
            break
        # Offset next row for hexagonal packing
        x = r_initial + r_initial * np.sqrt(3) / 2
        while x + r_initial <= 1.0:
            centers.append([x, y])
            x += r_initial * np.sqrt(3)
            if len(centers) >= n_circles:
                break
        y += r_initial * 1.5
        if len(centers) >= n_circles:
            break

    # Ensure we have exactly 26 circles
    centers = np.array(centers[:n_circles])
    radii = np.full(n_circles, r_initial)

    # 2. Optimization Setup
    # Combine centers and radii into a single vector for optimization
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]

    # Define objective function: Maximize sum of radii -> Minimize -sum
    def objective(variables):
        return -np.sum(variables[2::3])

    # Define constraints
    constraints = []
    bounds = []

    # Boundary constraints and non-negative radii
    for i in range(n_circles):
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        # r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+2]})
        
        # Bounds for variables to help optimizer
        # x in [0, 1]
        bounds.append((0.0, 1.0))
        # y in [0, 1]
        bounds.append((0.0, 1.0))
        # r >= 0
        bounds.append((0.0, 1.0))

    # Non-overlap constraints: dist(i, j) >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) >= ri + rj
    # (xi-xj)^2 + (yi-yj)^2 >= (ri + rj)^2
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })

    # Run optimization
    # SLSQP is suitable for constrained nonlinear optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if result.success:
        opt_variables = result.x
    else:
        # Fallback to initial guess if optimization fails
        opt_variables = x0

    # Extract results
    final_centers = np.zeros((n_circles, 2))
    final_radii = np.zeros(n_circles)
    for i in range(n_circles):
        final_centers[i, 0] = opt_variables[3*i]
        final_centers[i, 1] = opt_variables[3*i+1]
        final_radii[i] = opt_variables[3*i+2]

    sum_radii = np.sum(final_radii)
    return final_centers, final_radii, sum_radii
