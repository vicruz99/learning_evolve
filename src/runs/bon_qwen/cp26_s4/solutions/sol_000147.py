# sol_000147 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3a06727e) state=e60bb6e7 sum of radii=2.554312 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: 5x5 Grid
    # A 5x5 grid fits 25 circles of radius 0.1. We start there and add one more circle 
    # slightly larger to allow the optimizer to explore configurations.
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.102) # Start slightly above 0.1 to allow the optimizer to find valid spacing
    
    k = 0
    for row in range(5):
        for col in range(5):
            if k < n:
                # Grid points with small margins to allow optimization
                centers[k, 0] = (col + 0.5) / 5.0
                centers[k, 1] = (row + 0.5) / 5.0
                k += 1
                
    # Adjust the last circle if needed (though 25 loops covers 25, we have 26)
    # The above loop runs 25 times. We need to set the 26th circle (index 25)
    # We place it near the center with a smaller radius initially to avoid massive overlap
    if k < n:
        centers[25, 0] = 0.5
        centers[25, 1] = 0.5
        radii[25] = 0.01

    # 2. Define Objective Function (Negative sum to minimize)
    def objective(vars):
        # vars contains x1, y1, r1, x2, y2, r2, ...
        # We only optimize radii and centers.
        # Extract radii from every 3rd variable starting at index 2
        current_radii = vars[2::3]
        return -np.sum(current_radii)

    # 3. Define Constraints
    constraints = []
    
    # Box constraints and non-negative radii
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0, r >= 0
    for i in range(n):
        # x_i - r_i >= 0
        def c_box_x_min(index=i):
            def fun(vars):
                x = vars[index * 3]
                r = vars[index * 3 + 2]
                return x - r
            return fun
        constraints.append({'type': 'ineq', 'fun': c_box_x_min()})

        # 1 - x_i - r_i >= 0
        def c_box_x_max(index=i):
            def fun(vars):
                x = vars[index * 3]
                r = vars[index * 3 + 2]
                return 1.0 - x - r
            return fun
        constraints.append({'type': 'ineq', 'fun': c_box_x_max()})

        # y_i - r_i >= 0
        def c_box_y_min(index=i):
            def fun(vars):
                y = vars[index * 3 + 1]
                r = vars[index * 3 + 2]
                return y - r
            return fun
        constraints.append({'type': 'ineq', 'fun': c_box_y_min()})

        # 1 - y_i - r_i >= 0
        def c_box_y_max(index=i):
            def fun(vars):
                y = vars[index * 3 + 1]
                r = vars[index * 3 + 2]
                return 1.0 - y - r
            return fun
        constraints.append({'type': 'ineq', 'fun': c_box_y_max()})
        
        # r_i >= 0
        def c_r_nonneg(index=i):
            def fun(vars):
                return vars[index * 3 + 2]
            return fun
        constraints.append({'type': 'ineq', 'fun': c_r_nonneg()})

    # Overlap constraints: dist^2 >= (r_i + r_j)^2  => dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            def c_overlap(i=i, j=j):
                def fun(vars):
                    x1, y1, r1 = vars[i*3], vars[i*3+1], vars[i*3+2]
                    x2, y2, r2 = vars[j*3], vars[j*3+1], vars[j*3+2]
                    
                    dx = x1 - x2
                    dy = y1 - y2
                    dist_sq = dx*dx + dy*dy
                    rad_sum = r1 + r2
                    
                    return dist_sq - rad_sum**2
                return fun
            constraints.append({'type': 'ineq', 'fun': c_overlap()})

    # 4. Setup Initial State for Optimizer
    # Flatten centers and radii into a single vector
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[i * 3] = centers[i, 0]
        x0[i * 3 + 1] = centers[i, 1]
        x0[i * 3 + 2] = radii[i]

    # 5. Run Optimization
    try:
        res = minimize(objective, x0, method='SLSQP', constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        best_vars = res.x
    except Exception as e:
        # Fallback to initial if optimization fails
        best_vars = x0

    # 6. Extract Results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = best_vars[i * 3]
        final_centers[i, 1] = best_vars[i * 3 + 1]
        final_radii[i] = best_vars[i * 3 + 2]
        
    # Ensure non-negative radii (clamp small negatives to 0)
    final_radii = np.maximum(final_radii, 0)
    
    # Adjust centers if radii were clamped to satisfy bounds
    for i in range(n):
        r = final_radii[i]
        final_centers[i, 0] = np.clip(final_centers[i, 0], r, 1.0 - r)
        final_centers[i, 1] = np.clip(final_centers[i, 1], r, 1.0 - r)

    total_sum = np.sum(final_radii)
    
    return final_centers, final_radii, total_sum
