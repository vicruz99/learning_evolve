# sol_000029 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfc1b343) state=65279232 sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def generate_hexagonal_centers(n, square_size=1.0, margin=0.01):
    """
    Generates an initial set of circle centers in a hexagonal pattern.
    """
    centers = []
    row_idx = 0
    # Estimate rows based on square root of n
    cols = int(np.ceil(np.sqrt(n)))
    
    # Approximate spacing to fit within the square
    # Width constraint: 2r + (cols-1)*2r = 1 => r = 1/(2*cols) approx
    # Vertical spacing for hex: sqrt(3)*r
    r_est = 1.0 / (2 * cols)
    y_spacing = np.sqrt(3) * r_est
    
    y_pos = r_est + margin
    while len(centers) < n:
        x_spacing = 2 * r_est
        x_offset = r_est + margin + (row_idx % 2) * r_est
        
        # Calculate number of columns for this row to fit within square
        max_x = 1.0 - margin
        num_cols = 0
        curr_x = x_offset
        while curr_x <= max_x and len(centers) < n:
            centers.append([curr_x, y_pos])
            curr_x += x_spacing
            num_cols += 1
            
        y_pos += y_spacing
        row_idx += 1
        
        if y_pos + r_est > 1.0 - margin:
            # If we run out of vertical space, reset or break
            # For this heuristic, we just keep adding to last valid row or shrink
            break
            
    # If we have too many, trim; if too few, we might need to adjust logic, 
    # but hex generation usually works if parameters are tuned.
    # Fallback: if not enough, fill with random or grid
    while len(centers) < n:
        centers.append([np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)])
        
    return np.array(centers[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize centers
    centers_init = generate_hexagonal_centers(n)
    
    # 2. Initialize radii (start small to ensure validity)
    radii_init = np.full(n, 0.05)
    
    # 3. Define optimization function
    def objective(vars):
        # vars structure: [x0, y0, r0, x1, y1, r1, ...]
        return -np.sum(vars[2::3])  # Maximize sum of radii

    def constraints(vars):
        cons = []
        
        for i in range(n):
            idx = i * 3
            x, y, r = vars[idx], vars[idx+1], vars[idx+2]
            
            # Boundary constraints: x - r >= 0, x + r <= 1, etc.
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3] - v[i*3+2]})          # x >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i*3] - v[i*3+2]})   # 1 - x >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[i*3+1] - v[i*3+2]})       # y >= r
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[i*3+1] - v[i*3+2]}) # 1 - y >= r
            
            # Non-overlap constraints
            for j in range(i + 1, n):
                jdx = j * 3
                # dist^2 >= (r_i + r_j)^2
                # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
                def overlap(v, i=i, j=j):
                    xi, yi, ri = v[i*3], v[i*3+1], v[i*3+2]
                    xj, yj, rj = v[j*3], v[j*3+1], v[j*3+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    sum_r_sq = (ri + rj)**2
                    return dist_sq - sum_r_sq
                cons.append({'type': 'ineq', 'fun': overlap})
                
        return cons

    # 4. Prepare initial vector
    vars_init = []
    for i in range(n):
        vars_init.extend([centers_init[i, 0], centers_init[i, 1], radii_init[i]])
    vars_init = np.array(vars_init)

    # 5. Run optimization
    # Use SLSQP for non-linear constraints
    # Bounds for radii to be non-negative
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n # x, y in [0,1], r in [0, 0.5]
    
    try:
        result = minimize(objective, vars_init, method='SLSQP', 
                         bounds=bounds, constraints=constraints(),
                         options={'maxiter': 1000, 'ftol': 1e-9})
        
        vars_opt = result.x
    except Exception:
        # Fallback to initial guess if optimization fails
        vars_opt = vars_init

    # 6. Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i] = [vars_opt[i*3], vars_opt[i*3+1]]
        radii[i] = vars_opt[i*3+2]

    # 7. Final validation and slight adjustment for numerical stability
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    # Ensure centers are within [0, 1] adjusted for radius
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        centers[i, 0] = np.clip(x, r, 1.0 - r)
        centers[i, 1] = np.clip(y, r, 1.0 - r)

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Note: The provided validation function is read-only and will be called externally.
# We define run_packing as requested.
