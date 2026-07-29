# sol_000012 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b4d6f452) state=e34ea4a5 sum of radii=2.552137 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global constant for the number of circles
N_CIRCLES = 26

def init_packing():
    """
    Initialize circles in a hexagonal pattern with a small radius.
    This provides a valid starting point for the optimizer.
    """
    r_start = 0.04
    centers = []
    
    # Hexagonal packing parameters
    y_step = r_start * np.sqrt(3)
    x_step = 2 * r_start
    
    count = 0
    row = 0
    y = r_start
    n = N_CIRCLES
    
    # Fill rows in a hexagonal pattern
    while count < n:
        shift = 0 if row % 2 == 0 else r_start
        
        # Calculate how many circles fit in this row
        if shift > 0:
            num = int((1 - 2 * r_start - shift) / (2 * r_start)) + 1
        else:
            num = int((1 - 2 * r_start) / (2 * r_start)) + 1
            
        num = max(1, num)
        if count + num > n:
            num = n - count
            
        for k in range(num):
            x = shift + r_start + k * x_step
            centers.append([x, y])
            count += 1
            
        y += y_step
        row += 1
        
    centers_arr = np.array(centers)
    radii_arr = np.full(n, r_start)
    
    # Flatten to optimization variable vector: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_arr[i, 0]
        x0[3*i+1] = centers_arr[i, 1]
        x0[3*i+2] = radii_arr[i]
        
    # Add tiny random perturbation to break symmetry
    np.random.seed(42)
    x0 += np.random.randn(3 * n) * 1e-4
    
    return x0

def objective(x):
    """
    Objective function: maximize sum of radii.
    Since we minimize, we return the negative sum.
    """
    # Radii are at indices 2, 5, 8, ...
    radii = x[2::3]
    return -np.sum(radii)

def constraints(x):
    """
    Inequality constraints g(x) >= 0.
    Includes boundary constraints and pairwise non-overlap constraints.
    """
    n = N_CIRCLES
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    cons = []
    
    # Boundary constraints:
    # x_i >= r_i  => x_i - r_i >= 0
    # x_i <= 1 - r_i => 1 - x_i - r_i >= 0
    # Same for y
    cons.extend(cx - r)
    cons.extend(1 - cx - r)
    cons.extend(cy - r)
    cons.extend(1 - cy - r)
    
    # Overlap constraints:
    # distance^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d2 = dx*dx + dy*dy
            r_sum = r[i] + r[j]
            cons.append(d2 - r_sum*r_sum)
            
    return np.array(cons)

def run_packing():
    """
    Main function to run the optimization and return the packing.
    """
    n = N_CIRCLES
    x0 = init_packing()
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    
    # Options for SLSQP
    options = {'maxiter': 2000, 'ftol': 1e-12}
    
    best_x = x0
    best_val = objective(x0)
    
    # Lower and upper bounds for projection
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Run optimization with multiple restarts to improve robustness
    for k in range(3):
        if k > 0:
            # Perturb the best solution found so far
            x_curr = best_x + np.random.randn(3 * n) * 0.01
        else:
            x_curr = x0
            
        # Project back to bounds
        x_curr = np.clip(x_curr, lb, ub)
        
        # Run SLSQP
        res = minimize(objective, x_curr, method='SLSQP', bounds=bounds, 
                       constraints={'type': 'ineq', 'fun': constraints}, options=options)
        
        # Update best if successful and better
        if res.success:
            if res.fun < best_val:
                best_x = res.x
                best_val = res.fun
                
    # Extract results
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    
    return centers, radii, np.sum(radii)
