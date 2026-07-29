# sol_000204 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7f4d5c4f) state=b9687377 sum of radii=2.575397 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    
    # Helper function to compute constraints
    def compute_constraints(x_vars):
        # x_vars contains [x1, y1, r1, x2, y2, r2, ...]
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i] = [x_vars[3*i], x_vars[3*i+1]]
            radii[i] = x_vars[3*i+2]
        
        cons = []
        
        # Boundary constraints: 4 per circle
        # x - r >= 0  => x - r
        # 1 - (x + r) >= 0 => 1 - x - r
        # y - r >= 0 => y - r
        # 1 - (y + r) >= 0 => 1 - y - r
        for i in range(n):
            x = x_vars[3*i]
            y = x_vars[3*i+1]
            r = x_vars[3*i+2]
            
            cons.append(x - r)
            cons.append(1.0 - x - r)
            cons.append(y - r)
            cons.append(1.0 - y - r)
            
        # Overlap constraints: (dist)^2 >= (r_i + r_j)^2
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x_vars[3*i] - x_vars[3*j]
                dy = x_vars[3*i+1] - x_vars[3*j+1]
                r_sum = x_vars[3*i+2] + x_vars[3*j+2]
                
                dist_sq = dx*dx + dy*dy
                r_sum_sq = r_sum * r_sum
                
                cons.append(dist_sq - r_sum_sq)
                
        return np.array(cons)

    def objective(x_vars):
        # Maximize sum of radii -> Minimize negative sum
        radii_sum = sum(x_vars[3*i+2] for i in range(n))
        return -radii_sum

    # Initial Guess Construction
    # Hexagonal packing pattern: 5, 4, 5, 4, 5, 3 circles in rows
    # This totals 26 circles.
    row_counts = [5, 4, 5, 4, 5, 3]
    r_init = 0.08  # Initial radius, small enough to fit easily
    
    x0 = []
    row_y = r_init
    
    for i, count in enumerate(row_counts):
        # Hexagonal packing:
        # Even rows (0, 2, 4): start at r, step 2r
        # Odd rows (1, 3, 5): start at 2r (shifted by r), step 2r
        
        if i % 2 == 0:
            start_x = r_init
        else:
            start_x = 2 * r_init
            
        for k in range(count):
            x = start_x + k * (2 * r_init)
            # Clamp x to be within [0, 1] just in case, though logic ensures it for small r
            x = np.clip(x, 0, 1)
            
            # Add x, y, r to initial guess
            x0.append(x)
            x0.append(row_y)
            x0.append(r_init)
            
        row_y += r_init * np.sqrt(3)
        
    # Ensure we have exactly 26 circles (78 variables)
    if len(x0) != 78:
        # Fallback to random valid grid if count mismatch
        # This shouldn't happen with the counts defined
        pass

    x0 = np.array(x0)

    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])
        
    # Define constraints dictionary for SLSQP
    # SLSQP expects constraints in the form: fun(x) >= 0
    # We pass the function compute_constraints which returns an array of inequalities >= 0
    
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # Run optimization
    # Using SLSQP which supports bounds and nonlinear constraints
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
        
        if res.success or res.fun < -2.0: # If successful or reasonable objective
            final_centers = np.zeros((n, 2))
            final_radii = np.zeros(n)
            for i in range(n):
                final_centers[i] = [res.x[3*i], res.x[3*i+1]]
                final_radii[i] = res.x[3*i+2]
            
            # Basic sanity check and fix numerical errors
            for i in range(n):
                final_centers[i][0] = np.clip(final_centers[i][0], final_radii[i], 1.0 - final_radii[i])
                final_centers[i][1] = np.clip(final_centers[i][1], final_radii[i], 1.0 - final_radii[i])
                final_radii[i] = max(0.0, final_radii[i])
                
            sum_r = np.sum(final_radii)
            return final_centers, final_radii, sum_r

    except Exception as e:
        pass

    # Fallback solution if optimization fails
    # Return the initial configuration (scaled down if needed, but r=0.08 is valid)
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    row_y = r_init
    idx = 0
    for i, count in enumerate(row_counts):
        if i % 2 == 0:
            start_x = r_init
        else:
            start_x = 2 * r_init
            
        for k in range(count):
            x = start_x + k * (2 * r_init)
            x = np.clip(x, 0, 1)
            
            final_centers[idx] = [x, row_y]
            final_radii[idx] = r_init
            idx += 1
        row_y += r_init * np.sqrt(3)
        
    return final_centers, final_radii, np.sum(final_radii)
