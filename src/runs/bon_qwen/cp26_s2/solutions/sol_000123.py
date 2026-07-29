# sol_000123 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 840b35ba) state=c50680c0 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Returns a packing of 26 circles in a unit square [0,1]x[0,1] 
    that maximizes the sum of radii.
    """
    n_circles = 26
    
    # 1. Hexagonal Lattice Initialization
    # We aim for 6 rows. To fit 26 circles, we use a pattern like 5-4-5-4-5-4 (sum=27) 
    # and remove one, or just let the optimizer handle the exact count from a dense start.
    # Here we initialize 26 circles in 6 rows.
    
    rows = 6
    # Approximate radius for 6 rows hex packing: r ~ 1 / (2 + (rows-1)*sqrt(3)) ~ 0.094
    # Width constraint for 5 circles: 11r <= 1 => r <= 0.0909.
    init_r = 0.09
    
    centers = []
    radii = []
    count = 0
    
    for i in range(rows):
        y = init_r + i * (np.sqrt(3) * init_r)
        if count >= n_circles:
            break
            
        # Determine shift for hexagonal packing
        if i % 2 == 0:
            x_start = init_r
        else:
            x_start = 2 * init_r # Shifted by r
            
        # Calculate how many circles fit in this row
        # For a row with k circles starting at x_start, last center is x_start + (k-1)*2r
        # We need last center + r <= 1
        # x_start + (k-1)*2r + r <= 1 => x_start + r + (k-1)*2r <= 1
        # Let's just place circles until we hit boundary
        
        k = 0
        while count < n_circles:
            x = x_start + k * (2 * init_r)
            if x + init_r > 1.0 + 1e-5:
                break
            centers.append([x, y])
            radii.append(init_r)
            count += 1
            k += 1
            
    centers = np.array(centers)
    radii = np.array(radii)
    
    # 2. Optimization Function
    # We want to maximize sum(radii) -> minimize -sum(radii)
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Total 78 variables.
    
    def objective(vars_flat):
        # vars_flat shape (78,)
        radii_curr = vars_flat[2::3] # Every 3rd element starting at index 2
        return -np.sum(radii_curr)
    
    def constraints(vars_flat):
        n = len(vars_flat) // 3
        centers_curr = np.zeros((n, 2))
        radii_curr = np.zeros(n)
        
        for i in range(n):
            centers_curr[i] = [vars_flat[3*i], vars_flat[3*i+1]]
            radii_curr[i] = vars_flat[3*i+2]
            
        con_list = []
        
        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        # x - r >= 0 => x - r
        # 1 - r - x >= 0 => 1 - x - r
        for i in range(n):
            con_list.append(vars_flat[3*i] - vars_flat[3*i+2])          # x - r >= 0
            con_list.append(1.0 - vars_flat[3*i] - vars_flat[3*i+2])    # 1 - x - r >= 0
            con_list.append(vars_flat[3*i+1] - vars_flat[3*i+2])        # y - r >= 0
            con_list.append(1.0 - vars_flat[3*i+1] - vars_flat[3*i+2])  # 1 - y - r >= 0
            
        # Overlap constraints: dist >= r1 + r2
        # (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
        # (x1-x2)^2 + (y1-y2)^2 - (r1+r2)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = vars_flat[3*i] - vars_flat[3*j]
                dy = vars_flat[3*i+1] - vars_flat[3*j+1]
                r_sum = vars_flat[3*i+2] + vars_flat[3*j+2]
                con_list.append(dx**2 + dy**2 - r_sum**2)
                
        return np.array(con_list)
        
    # Initial guess
    x0 = np.zeros(3 * n_circles)
    for i in range(n_circles):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds: [0, 1] for x, y; [0, 0.5] for r (r cannot exceed 0.5)
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': constraints}, options={'maxiter': 1000, 'ftol': 1e-9})
    
    # Extract results
    best_vars = res.x
    final_centers = np.zeros((n_circles, 2))
    final_radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        final_centers[i, 0] = best_vars[3*i]
        final_centers[i, 1] = best_vars[3*i+1]
        final_radii[i] = best_vars[3*i+2]
        
    # Clip radii to be non-negative just in case of numerical noise
    final_radii = np.maximum(final_radii, 0.0)
    
    # Adjust centers if needed to be strictly inside (though constraints should handle it)
    # Re-validating and fixing slight boundary violations if any due to optimizer tolerance
    for i in range(n_circles):
        r = final_radii[i]
        # x
        if final_centers[i, 0] < r: final_centers[i, 0] = r
        if final_centers[i, 0] > 1.0 - r: final_centers[i, 0] = 1.0 - r
        # y
        if final_centers[i, 1] < r: final_centers[i, 1] = r
        if final_centers[i, 1] > 1.0 - r: final_centers[i, 1] = 1.0 - r
        
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
