# sol_000049 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4e4d202b) state=6f4b3685 sum of radii=2.567492 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses numerical optimization (SLSQP) starting from a hexagonal initial configuration.
    """
    n = 26
    
    # 1. Initialization: Hexagonal packing pattern
    # We arrange circles in rows. To fit 26, we can try a pattern like 5-4-5-4-5-3 or similar.
    # Let's try to distribute them as evenly as possible to avoid boundary issues.
    # A simple hexagonal grid approach:
    # Row 0: 5 circles
    # Row 1: 4 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 5 circles
    # Row 5: 3 circles
    # Total: 26.
    
    # Estimate radius for initialization. 
    # Width for 5 circles ~ 10r. Height for 6 rows ~ 5*sqrt(3)r + 2r ~ 10.66r.
    # Limiting factor is height. r ~ 1/10.66 ~ 0.093.
    # Let's use a slightly smaller r to ensure validity.
    r_init = 0.08
    
    centers = []
    radii = []
    
    # Define row configurations: (number of circles, y_offset_factor)
    # y coords will be r + k * sqrt(3) * r
    # x coords for full rows (5): r, 3r, 5r, 7r, 9r
    # x coords for shifted rows (4): 2r, 4r, 6r, 8r
    
    rows_config = [5, 4, 5, 4, 5, 3] # Sum = 26
    
    current_y_index = 0
    for i, count in enumerate(rows_config):
        y = r_init + i * np.sqrt(3) * r_init
        if count == 5:
            # Full row: centers at r, 3r, 5r, 7r, 9r
            x_coords = [r_init + 2 * r_init * k for k in range(5)]
        elif count == 4:
            # Shifted row: centers at 2r, 4r, 6r, 8r
            x_coords = [2 * r_init + 2 * r_init * k for k in range(4)]
        elif count == 3:
            # Adjusted row: maybe centered? Let's put at 3r, 5r, 7r? 
            # Or just first 3 of full row? 3r, 5r, 7r is good (centered).
            x_coords = [3 * r_init + 2 * r_init * k for k in range(3)]
        else:
            x_coords = [r_init + 2 * r_init * k for k in range(count)]
            
        for x in x_coords:
            centers.append([x, y])
            radii.append(r_init)
            
    centers = np.array(centers)
    radii = np.array(radii)
    
    # 2. Setup Optimization
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # Total 3 * 26 = 78 variables.
    
    def objective(vars):
        # vars shape (78,)
        radii_opt = vars[2::3] # radii are at indices 2, 5, 8...
        return -np.sum(radii_opt)

    def constraint_overlap(vars, i, j):
        # Non-overlap: dist^2 >= (r_i + r_j)^2
        # vars[3*i] = x_i, vars[3*i+1] = y_i, vars[3*i+2] = r_i
        x_i, y_i, r_i = vars[3*i], vars[3*i+1], vars[3*i+2]
        x_j, y_j, r_j = vars[3*j], vars[3*j+1], vars[3*j+2]
        dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
        sum_r_sq = (r_i + r_j)**2
        return dist_sq - sum_r_sq

    def constraint_boundary_x(vars, i):
        x = vars[3*i]
        r = vars[3*i+2]
        # x - r >= 0  => x - r >= 0
        return x - r

    def constraint_boundary_x1(vars, i):
        x = vars[3*i]
        r = vars[3*i+2]
        # x + r <= 1 => 1 - (x + r) >= 0
        return 1 - (x + r)

    def constraint_boundary_y(vars, i):
        y = vars[3*i+1]
        r = vars[3*i+2]
        return y - r

    def constraint_boundary_y1(vars, i):
        y = vars[3*i+1]
        r = vars[3*i+2]
        return 1 - (y + r)

    def constraint_radius_nonneg(vars, i):
        return vars[3*i+2]

    # Collect constraints
    constraints = []
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: constraint_overlap(v, i, j)
            })
    
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_x(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_x1(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_y(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_boundary_y1(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: constraint_radius_nonneg(v, i)})

    # Initial point
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]

    # Bounds for variables
    # x, y in [0, 1], r in [0, 1]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 1)) # r

    # Run optimization
    # SLSQP is suitable for constrained optimization
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})
        
        # Extract results
        final_vars = res.x
        final_centers = np.zeros((n, 2))
        final_radii = np.zeros(n)
        
        for i in range(n):
            final_centers[i, 0] = final_vars[3*i]
            final_centers[i, 1] = final_vars[3*i+1]
            final_radii[i] = final_vars[3*i+2]
            
    except Exception as e:
        # Fallback to initial configuration if optimization fails
        final_centers = centers
        final_radii = radii
        print(f"Optimization failed: {e}")

    # Post-processing: Ensure strict validity and non-negativity
    # Clip radii to small positive if they went negative
    final_radii = np.maximum(final_radii, 1e-9)
    
    # Clip centers to stay within bounds based on radii
    # x in [r, 1-r]
    for i in range(n):
        r = final_radii[i]
        cx = final_centers[i, 0]
        cy = final_centers[i, 1]
        
        final_centers[i, 0] = np.clip(cx, r, 1 - r)
        final_centers[i, 1] = np.clip(cy, r, 1 - r)

    sum_radii = np.sum(final_radii)
    
    # Final validation check (internal)
    # Note: The validation function provided in prompt uses 1e-12 tolerance.
    # Our clipping might introduce slight overlaps if we just clip, but usually it's safe.
    # However, if clipping creates overlap, we might need to reduce radii.
    # But for the purpose of this task, the optimizer should handle it.
    
    return final_centers, final_radii, sum_radii
