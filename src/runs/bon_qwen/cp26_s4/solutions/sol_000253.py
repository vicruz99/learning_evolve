# sol_000253 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 30e75f73) state=410437c9 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def calculate_sum(radii):
    return np.sum(radii)

def constraint_boundary(params, n):
    # Extract x, y, and radii from the flattened parameter array
    centers = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    cons = []
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # x - r >= 0
        cons.append(x - r)
        # x + r <= 1 => 1 - (x + r) >= 0
        cons.append(1 - (x + r))
        # y - r >= 0
        cons.append(y - r)
        # y + r <= 1 => 1 - (y + r) >= 0
        cons.append(1 - (y + r))
    return np.array(cons)

def constraint_overlap(params, n):
    centers = params[:2 * n].reshape(n, 2)
    radii = params[2 * n:]
    
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            # dist >= r_i + r_j => dist - (r_i + r_j) >= 0
            cons.append(dist - (radii[i] + radii[j]))
    return np.array(cons)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- Initialization ---
    # Hexagonal packing layout
    rows = 6
    cols = 5
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Base radius for 5x5 grid is 0.1. Hex packing allows slightly larger, 
    # but we start conservatively to ensure feasible initial state.
    r_init = 0.08 
    dx = 2 * r_init
    dy = r_init * np.sqrt(3)
    
    # Add vertical offset to center the grid in the square
    y_offset = 0.5 - (rows - 1) * dy / 2
    
    count = 0
    for r in range(rows):
        # Offset every other row for hexagonal lattice
        x_offset = 0.5 * dx if r % 2 == 1 else 0
        for c in range(cols):
            if count < n:
                centers[count, 0] = 0.1 + c * dx + x_offset
                centers[count, 1] = y_offset + r * dy
                radii[count] = r_init
                count += 1
                
    # --- Optimization ---
    # Flatten parameters: [x0, y0, x1, y1, ..., rn-1]
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Define bounds (0 <= x, y <= 1; 0 <= r)
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0)])
    for _ in range(n):
        bounds.append((0.0, 1.0))
        
    # Define constraints
    cons = [
        {'type': 'ineq', 'fun': lambda p: constraint_boundary(p, n)},
        {'type': 'ineq', 'fun': lambda p: constraint_overlap(p, n)}
    ]
    
    # Objective: Maximize sum of radii (minimize negative sum)
    def objective(params):
        r_vals = params[2 * n:]
        return -np.sum(r_vals)
        
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                      options={'maxiter': 1000, 'disp': False})
    
    if result.success:
        final_centers = result.x[:2 * n].reshape(n, 2)
        final_radii = result.x[2 * n:]
        sum_r = np.sum(final_radii)
        return final_centers, final_radii, sum_r
    else:
        # Fallback to initial configuration if optimization fails
        return centers, radii, np.sum(radii)
