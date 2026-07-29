# sol_000214 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dddb8969) state=8d05956e sum of radii=2.571295 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    Uses a hexagonal grid initialization and SLSQP optimization.
    """
    n_circles = 26
    
    # 1. Initialize centers and radii using a dense hexagonal grid
    centers = np.zeros((n_circles, 2))
    radii = np.full(n_circles, 0.08) # Safe initial radius
    
    r_init = 0.08
    y = r_init
    count = 0
    
    while count < n_circles:
        # Alternate x-start for hexagonal packing
        row_num = int(y / (r_init * math.sqrt(3)))
        x_start = r_init if (row_num % 2 == 0) else 2 * r_init
        
        x = x_start
        while x <= 1 - r_init:
            if count < n_circles:
                centers[count] = [x, y]
                count += 1
            x += 2 * r_init
        y += r_init * math.sqrt(3)

    # Fill any remaining spots (should not happen with r=0.08)
    for i in range(count, n_circles):
        centers[i] = [0.5, 0.5]
        radii[i] = 0.01

    # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Define bounds for [x, y, r]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n_circles
    
    # Constraint function returning array of values (must be >= 0)
    def eval_constraints(v):
        vals = np.zeros(4 * n_circles + (n_circles * (n_circles - 1)) // 2)
        idx = 0
        
        # Boundary constraints (x-r>=0, 1-x-r>=0, y-r>=0, 1-y-r>=0)
        for i in range(n_circles):
            ix, iy, ir = 3*i, 3*i+1, 3*i+2
            x, y, r = v[ix], v[iy], v[ir]
            vals[idx] = x - r; idx += 1
            vals[idx] = 1.0 - x - r; idx += 1
            vals[idx] = y - r; idx += 1
            vals[idx] = 1.0 - y - r; idx += 1
            
        # Non-overlap constraints (dist^2 - (r_i+r_j)^2 >= 0)
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                ix_i, iy_i, ir_i = 3*i, 3*i+1, 3*i+2
                ix_j, iy_j, ir_j = 3*j, 3*j+1, 3*j+2
                
                dx = v[ix_i] - v[ix_j]
                dy = v[iy_i] - v[iy_j]
                dr = v[ir_i] + v[ir_j]
                
                vals[idx] = (dx*dx + dy*dy) - (dr*dr)
                idx += 1
                
        return vals

    # Objective: Maximize sum of radii -> Minimize negative sum
    def objective(v):
        return -np.sum(v[2::3])

    # Run optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                   constraints={'type': 'ineq', 'fun': eval_constraints},
                   options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})
    
    # Extract final results
    final_centers = res.x.reshape(n_circles, 3)[:, :2]
    final_radii = res.x.reshape(n_circles, 3)[:, 2]
    total_sum = np.sum(final_radii)
    
    return final_centers, final_radii, total_sum
