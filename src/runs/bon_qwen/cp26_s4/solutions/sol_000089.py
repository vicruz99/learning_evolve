# sol_000089 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f395aea4) state=d234bbdf sum of radii=2.585427 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def hexagonal_grid_init(n, r=0.01):
    """
    Generates a hexagonal grid initialization for n circles.
    """
    centers = []
    r_start = r
    row_count = 0
    col_count = 0
    sqrt3 = math.sqrt(3)
    
    while len(centers) < n:
        # Hexagonal spacing
        y = r_start + row_count * sqrt3 * r_start
        x_start = r_start if row_count % 2 == 0 else r_start + r_start
        
        # Check bounds
        if y + r_start > 1:
            row_count += 1
            continue
            
        x = x_start
        while x + r_start <= 1 and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_start
        
        row_count += 1
        
    # Return first n centers
    return np.array(centers[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- Objective and Constraints for Optimization ---
    
    def objective(vars):
        # vars contains [x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
        radii = vars[2::3]
        return -np.sum(radii) # Minimize negative sum

    def boundary_constraints(vars):
        # Each circle must be inside [0,1]x[0,1]
        # x >= r, x <= 1-r => x-r >= 0, x+r <= 1
        # y >= r, y <= 1-r => y-r >= 0, y+r <= 1
        cons = []
        for i in range(n):
            x = vars[3*i]
            y = vars[3*i+1]
            r = vars[3*i+2]
            cons.append(x - r)       # x >= r
            cons.append(1 - (x + r)) # x <= 1-r
            cons.append(y - r)       # y >= r
            cons.append(1 - (y + r)) # y <= 1-r
        return cons

    def non_overlap_constraints(vars):
        # dist_ij >= ri + rj => dist_ij^2 >= (ri+rj)^2
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
                xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
                
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                sum_r = ri + rj
                
                cons.append(dist_sq - sum_r**2)
        return cons

    # --- Initialization and Optimization Loop ---
    
    best_sum_radii = -1.0
    best_solution = None
    
    # Try multiple random perturbations of a hexagonal grid
    # Start with a reasonable radius estimate. 
    # For n=26, a grid of radius ~0.08 fits easily.
    initial_r = 0.08
    
    for attempt in range(10):
        # Generate initial positions
        centers = hexagonal_grid_init(n, initial_r)
        
        # Add some noise to break symmetry
        noise = np.random.uniform(-0.01, 0.01, size=centers.shape)
        centers = np.clip(centers + noise, 0.05, 0.95)
        
        # Initial radii
        radii = np.full(n, initial_r * 0.9) # Start slightly smaller
        
        # Construct initial variables vector
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = radii[i]
            
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for _ in range(n):
            bounds.extend([
                (0, 1),   # x
                (0, 1),   # y
                (1e-6, 0.5) # r
            ])
            
        # Constraints
        # SLSQP expects functions returning values >= 0
        # We wrap them in dictionaries or lists of dicts
        cons_list = []
        
        # Boundary constraints
        def b_cons(x):
            return boundary_constraints(x)
        # We need to split into individual constraints or pass as array? 
        # SLSQP can handle array of constraints if type is 'ineq'
        
        # Non-overlap constraints
        def no_cons(x):
            return non_overlap_constraints(x)

        constraints = [
            {'type': 'ineq', 'fun': b_cons},
            {'type': 'ineq', 'fun': no_cons}
        ]
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'ftol': 1e-9, 'maxiter': 2000})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_solution = res.x.copy()
                    
        except Exception:
            continue

    # --- Post-Processing / Extraction ---
    
    if best_solution is not None:
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = best_solution[3*i]
            centers[i, 1] = best_solution[3*i+1]
            radii[i] = best_solution[3*i+2]
    else:
        # Fallback to a simple grid if optimization fails completely
        # 5x5 grid plus 1 in center? No, 26 circles.
        # Just a 5x5 grid of 25 circles and 1 tiny one?
        # Better: 5x5 grid of radius 0.1, sum=2.5.
        # Let's try to fit 26 in a grid-like pattern.
        # 6, 5, 6, 5, 4 pattern?
        # Just return a valid solution.
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        idx = 0
        # Simple grid packing
        cols = 6
        rows = 5
        step_x = 1.0 / (cols + 1)
        step_y = 1.0 / (rows + 1)
        for r in range(rows):
            for c in range(cols):
                if idx < n:
                    centers[idx] = [step_x * (c + 1), step_y * (r + 1)]
                    # Estimate radius based on min distance to neighbors/boundaries
                    # Rough estimate
                    radii[idx] = min(step_x/2, step_y/2) * 0.95
                    idx += 1
        best_sum_radii = np.sum(radii)

    return centers, radii, best_sum_radii
