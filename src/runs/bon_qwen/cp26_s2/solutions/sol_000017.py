# sol_000017 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfb3fe63) state=13f8788b sum of radii=2.592693 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    
    # Helper function to compute distance
    def dist(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    # Define objective: maximize sum of radii -> minimize negative sum
    def objective(vars):
        # vars shape: (78,) -> (26, 3)
        x = vars.reshape(n_circles, 3)
        # x[:, 0] = x_coord, x[:, 1] = y_coord, x[:, 2] = radius
        return -np.sum(x[:, 2])

    # Define constraints
    # 1. Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    # 2. Non-overlap constraints: r_i + r_j <= dist(c_i, c_j)
    
    # We will use a penalty approach or explicit constraints. 
    # Explicit constraints are safer for validity.
    
    constraints = []
    
    # Boundary constraints
    for i in range(n_circles):
        # r <= x
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[i*3 + 2] - v[i*3 + 0]  # r - x <= 0 -> x - r >= 0? No.
            # Wait, scipy constraint: fun(x) >= 0 for 'ineq'.
            # We need x - r >= 0 => r <= x.
            # So fun = x - r.
        })
        # Actually, let's rewrite logic clearly.
        # Constraint: x - r >= 0
        # Constraint: 1 - x - r >= 0
        # Constraint: y - r >= 0
        # Constraint: 1 - y - r >= 0
        pass

    # Re-define constraints list properly
    cons_list = []
    
    for i in range(n_circles):
        idx_x = i * 3 + 0
        idx_y = i * 3 + 1
        idx_r = i * 3 + 2
        
        # x - r >= 0
        cons_list.append({'type': 'ineq', 'fun': lambda v, idx_x=idx_x, idx_r=idx_r: v[idx_x] - v[idx_r]})
        # 1 - x - r >= 0
        cons_list.append({'type': 'ineq', 'fun': lambda v, idx_x=idx_x, idx_r=idx_r: 1.0 - v[idx_x] - v[idx_r]})
        # y - r >= 0
        cons_list.append({'type': 'ineq', 'fun': lambda v, idx_y=idx_y, idx_r=idx_r: v[idx_y] - v[idx_r]})
        # 1 - y - r >= 0
        cons_list.append({'type': 'ineq', 'fun': lambda v, idx_y=idx_y, idx_r=idx_r: 1.0 - v[idx_y] - v[idx_r]})
        
        # r >= 0 (handled by bounds, but adding constraint for safety? Bounds are better)
        
    # Overlap constraints: dist(c_i, c_j) - (r_i + r_j) >= 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            idx_x_i, idx_y_i, idx_r_i = i*3, i*3+1, i*3+2
            idx_x_j, idx_y_j, idx_r_j = j*3, j*3+1, j*3+2
            
            # Constraint: sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
            def make_constraint(ii, jj, x1, y1, r1, x2, y2, r2):
                def constraint(v):
                    xi, yi = v[x1], v[y1]
                    xj, yj = v[x2], v[y2]
                    ri, rj = v[r1], v[r2]
                    d = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    return d - ri - rj
                return constraint
            
            cons_list.append({'type': 'ineq', 'fun': make_constraint(i, j, idx_x_i, idx_y_i, idx_r_i, idx_x_j, idx_y_j, idx_r_j)})

    # Bounds for variables
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1) for _ in range(n_circles * 3)]
    # Tighten radius bound? r cannot be > 0.5.
    for i in range(n_circles):
        bounds[i*3 + 2] = (0, 0.5)

    # Initial Guess
    # Generate a hexagonal-like grid
    # We want to place 26 circles.
    # Try to fit them in a pattern.
    # Rows: 5, 6, 5, 6, 4 = 26?
    # Let's try to generate points and then let optimizer fix them.
    
    np.random.seed(42)
    
    # Strategy: Place centers on a grid, then perturb.
    # A 5x5 grid gives 25 points. Add one in center?
    # Or random.
    
    # Let's create a dense random initialization
    x0 = np.random.rand(n_circles, 3)
    # Scale x, y to be well inside [0,1] initially?
    x0[:, 0] = 0.1 + 0.8 * x0[:, 0] # Map to [0.1, 0.9]
    x0[:, 1] = 0.1 + 0.8 * x0[:, 1]
    x0[:, 2] = 0.05 # Initial radius small
    
    # Better initial guess: Hexagonal packing
    # Approximate packing of 26 circles.
    # Let's try to arrange them in rows.
    # Row lengths: 5, 6, 5, 6, 4 (Total 26)
    # This might be tight.
    # Let's just use a random grid for robustness.
    
    # Actually, a structured grid is better for convergence.
    # Let's place them in a 6x5 grid (30 points) and pick 26?
    # Or just 26 points on a grid.
    
    points = []
    # Try to fill square with a grid
    # 5 columns, 5 rows = 25.
    # Add one more.
    
    # Let's generate 26 points uniformly
    grid_x = np.linspace(0.1, 0.9, 6) # 6 points?
    # Maybe just random is fine with good optimizer.
    
    # Let's try a specific layout:
    # 4 corners, 4 edges mid, 9 center grid?
    # Just use the random guess, it's safer against bad topology assumptions.
    
    x0_flat = x0.flatten()
    
    # Run optimization
    # SLSQP is good for constrained optimization
    try:
        result = minimize(objective, x0_flat, method='SLSQP', bounds=bounds, constraints=cons_list, 
                         options={'maxiter': 1000, 'ftol': 1e-9})
        
        if result.success:
            best_vars = result.x
        else:
            # If failed, maybe try a few random restarts?
            # But for now, take the result or fallback
            best_vars = x0_flat
    except Exception:
        best_vars = x0_flat

    # Post-processing: Ensure validity and clean up
    # Sometimes numerical errors leave tiny overlaps.
    # We can shrink radii slightly to ensure validity.
    
    centers = best_vars.reshape(n_circles, 3)[:, :2]
    radii = best_vars.reshape(n_circles, 3)[:, 2]
    
    # Check and fix overlaps by shrinking radii if necessary
    # This is a safety measure.
    # If overlap, reduce both radii.
    # Iterative fix
    for _ in range(10):
        changed = False
        for i in range(n_circles):
            # Boundary
            r_i = radii[i]
            x_i, y_i = centers[i]
            
            # Check boundaries
            max_r = min(x_i, 1-x_i, y_i, 1-y_i)
            if radii[i] > max_r + 1e-9:
                radii[i] = max_r
                changed = True
            
            # Check overlaps
            for j in range(i + 1, n_circles):
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                sum_r = radii[i] + radii[j]
                if sum_r > d + 1e-9:
                    # Overlap detected. Shrink both equally.
                    overlap = (sum_r - d) / 2.0 + 1e-6
                    radii[i] = max(0, radii[i] - overlap)
                    radii[j] = max(0, radii[j] - overlap)
                    changed = True
                    # Re-check boundary after shrinking? 
                    # Shrinking is always safe for boundary.
        if not changed:
            break
            
    # Recalculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, float(sum_radii)
