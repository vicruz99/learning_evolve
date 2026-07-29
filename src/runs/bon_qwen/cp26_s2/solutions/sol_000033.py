# sol_000033 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state eaaa636a) state=070755a6 sum of radii=2.476078 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_initial_packing(n=26):
    """
    Generates an initial valid packing of n circles using a hexagonal lattice pattern.
    Returns centers (n, 2) and radii (n,).
    """
    centers = []
    radii = []
    
    # Heuristic to distribute n circles into rows
    # Try to fit roughly sqrt(n) rows, alternating counts
    # Approximate width 1.0, height 1.0
    # Hexagonal packing vertical spacing: sqrt(3)/2 * 2r = sqrt(3)*r
    # Horizontal spacing: 2r
    # Let's assume r ~ 0.09 initially.
    # Width allows ~ 1/(2*0.09) ~ 5.5 circles per row.
    # Height allows ~ 1/(sqrt(3)*0.09) ~ 6.4 rows.
    
    # Let's try to build rows until we have n circles.
    # Row 0: 5 circles? 
    # Actually, let's just generate a grid and pick n points, 
    # but shifted to hexagonal.
    
    r_init = 0.07 # Safe radius
    cols = 6
    rows = 5
    
    # Generate hexagonal grid points
    grid_centers = []
    for r in range(rows):
        for c in range(cols):
            x = (c + 0.5) * (1.0 / cols) + (0.5 * (1.0 / cols)) * (r % 2)
            y = (r + 0.5) * (1.0 / rows)
            # Adjust to fit in [0,1] with margin r_init
            # Actually, just place them in [0,1] and let optimizer push them.
            # To be safe, scale to fit inside with margin.
            x = (c + 0.5 + 0.5 * (r % 2)) * (1.0 / (cols + 0.5))
            y = (r + 0.5) * (1.0 / rows)
            
            if 0 <= x <= 1 and 0 <= y <= 1:
                grid_centers.append([x, y])
    
    # Sort and pick top n
    grid_centers = grid_centers[:n]
    
    # If we don't have enough, fill with random valid points?
    # With 6x5=30 points, we have enough.
    
    centers = np.array(grid_centers)
    radii = np.full(n, r_init)
    
    return centers, radii

def objective_and_constraints(vars, n):
    """
    Computes objective and constraints for the optimizer.
    vars: flattened array [x1, y1, r1, x2, y2, r2, ...]
    """
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    obj = -np.sum(radii)
    
    # Constraints
    cons = []
    
    # Boundary constraints: r <= x <= 1-r  => x-r >= 0, 1-x-r >= 0
    # y-r >= 0, 1-y-r >= 0
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        cons.append(x - r)
        cons.append(1.0 - x - r)
        cons.append(y - r)
        cons.append(1.0 - y - r)
        
    # Overlap constraints: dist >= r1 + r2
    # (x1-x2)^2 + (y1-y2)^2 >= (r1+r2)^2
    # We use linearized constraint for SLSQP? No, quadratic is fine but SLSQP handles general non-linear.
    # To ensure positivity, we use dist - (r1+r2) >= 0.
    # Using squared distance avoids sqrt but is (dist^2 - sum_r^2) >= 0 which is different.
    # dist >= sum_r  <=> dist - sum_r >= 0.
    # Let's use dist - sum_r.
    
    # Vectorized calculation for efficiency? 
    # Loop is simpler to code for SLSQP constraint list, but slow.
    # Let's try to be efficient.
    
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            cons.append(dist - (radii[i] + radii[j]))
            
    return obj, np.array(cons)

def run_packing():
    n = 26
    centers, radii = get_initial_packing(n)
    
    # Flatten variables
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Optimization
    # We use SLSQP. 
    # To handle constraints, we pass a function that returns the constraint values >= 0.
    # But SLSQP expects a list of constraint dicts or a single constraint dict returning array.
    # Let's use a function that returns array.
    
    def constraints_func(vars):
        _, cons = objective_and_constraints(vars, n)
        return cons

    # Define constraints for scipy
    # type 'ineq' means >= 0
    cons_dict = {'type': 'ineq', 'fun': constraints_func}
    
    # Try to maximize sum of radii
    # Maximize sum(r) <=> Minimize -sum(r)
    
    # We need a function that returns just the objective for minimize
    def objective_func(vars):
        obj, _ = objective_and_constraints(vars, n)
        return obj

    # Initial check
    try:
        res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons_dict, 
                       options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
        x_opt = res.x
    except Exception as e:
        # Fallback if optimization fails
        x_opt = x0

    centers_opt = x_opt[:2*n].reshape(n, 2)
    radii_opt = x_opt[2*n:]
    
    # Ensure radii are non-negative (clip just in case)
    radii_opt = np.maximum(radii_opt, 0.0)
    
    # Recalculate sum
    sum_radii = np.sum(radii_opt)
    
    # Validate internally (optional but good for debugging)
    # If validation fails, we might need to adjust, but we trust optimizer with constraints.
    
    return centers_opt, radii_opt, sum_radii
