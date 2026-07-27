# sol_000052 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b794a107) state=c493afef sum of radii=2.500192 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Hexagonal packing initialization
    # We start with a valid configuration of 26 circles to help the optimizer.
    # A hexagonal lattice is denser than a square grid.
    r_init = 0.09
    row_height = np.sqrt(3) * r_init
    col_width = 2 * r_init
    
    idx = 0
    row_idx = 0
    
    while idx < n:
        y = r_init + row_idx * row_height
        
        # Shift for staggered rows to create hexagonal packing
        shift = 0.0
        if row_idx % 2 == 1:
            shift = r_init
        
        # Calculate valid x range for centers: [r_init, 1 - r_init]
        start_x = r_init + shift
        
        # If the shifted start is out of bounds, we can't place circles in this row with this shift.
        # However, with r=0.09, 1-r_init = 0.91, shift=0.09, start_x=0.18, which is valid.
        if start_x > 1.0 - r_init + 1e-9:
            # Move to next row if row is invalid (should not happen with these params)
            row_idx += 1
            continue
            
        # Calculate max circles that fit in the row
        # start_x + (count-1)*col_width <= 1 - r_init
        max_count = int((1.0 - r_init - start_x) / col_width) + 1
        if max_count < 1:
            max_count = 1
            
        count = min(max_count, n - idx)
        
        for col in range(count):
            x = start_x + col * col_width
            centers[idx] = [x, y]
            radii[idx] = r_init
            idx += 1
            
        row_idx += 1
        
    # Flatten variables for optimization: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    
    cons = []
    
    # 1. Boundary constraints: circles must be inside [0,1]x[0,1]
    def boundary_constraints(x):
        vals = []
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            vals.extend([
                xi - ri,          # x - r >= 0
                1.0 - xi - ri,    # x + r <= 1
                yi - ri,          # y - r >= 0
                1.0 - yi - ri     # y + r <= 1
            ])
        return np.array(vals)
    
    cons.append({'type': 'ineq', 'fun': boundary_constraints})
    
    # 2. Overlap constraints: distance between centers >= sum of radii
    def overlap_constraints(x):
        vals = []
        for i in range(n):
            xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
            for j in range(i + 1, n):
                xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
                dist = np.hypot(xi - xj, yi - yj)
                vals.append(dist - ri - rj)
        return np.array(vals)
    
    cons.append({'type': 'ineq', 'fun': overlap_constraints})
    
    # Objective function: Maximize sum of radii -> Minimize negative sum
    def objective(x):
        return -np.sum(x[2::3])
        
    # Run optimization using SLSQP
    # SLSQP handles bounds and non-linear inequality constraints
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
    
    # Extract solution
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i] = [res.x[3*i], res.x[3*i+1]]
        final_radii[i] = res.x[3*i+2]
        
    return final_centers, final_radii, np.sum(final_radii)
