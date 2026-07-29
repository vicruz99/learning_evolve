# sol_000069 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3353d097) state=f56d836d sum of radii=2.635932 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal initialization and SLSQP optimization.
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal Grid
    # We aim for a configuration that fits 26 circles. 
    # A 5-row arrangement works well. Distribution: 5, 6, 5, 5, 5 circles.
    # This sums to 26.
    # We start with a conservative radius to ensure feasibility.
    r_init = 0.08
    rows_counts = [5, 6, 5, 5, 5]
    
    centers_init = []
    radii_init = []
    
    # Vertical spacing for hexagonal packing is r * sqrt(3)
    # We center the rows vertically.
    # Total height for 5 rows with spacing s_v is 2*r + 4*s_v.
    # Let's approximate s_v = r * sqrt(3).
    # We can adjust vertical positions later, but let's spread them evenly in [r, 1-r]
    # to give the optimizer room.
    
    y_positions = np.linspace(r_init, 1 - r_init, 5)
    
    row_idx = 0
    circle_idx = 0
    for i, count in enumerate(rows_counts):
        y = y_positions[i]
        # Horizontal spacing 2*r. Center the row horizontally.
        # Width needed for 'count' circles is (count - 1) * 2*r + 2*r = count * 2*r ?
        # Actually, span is (count-1)*2r. Plus margins r on each side.
        # So x ranges from r to 1-r.
        # Points: r, r+2r, ..., r+(count-1)2r.
        # Check if r + (count-1)2r <= 1-r => count*2r <= 1.
        # For count=6, 12r <= 1 => r <= 0.0833. Our r_init=0.08 is safe.
        
        x_positions = np.linspace(r_init, 1 - r_init, count)
        
        for x in x_positions:
            centers_init.append([x, y])
            radii_init.append(r_init)
            circle_idx += 1
        row_idx += 1
            
    centers_init = np.array(centers_init)
    radii_init = np.array(radii_init)
    
    # 2. Setup Optimization
    # Variables: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    # Total 78 variables.
    
    def objective(vars):
        # vars shape (78,)
        r = vars[2::3]
        return -np.sum(r) # Maximize sum of radii => Minimize negative sum

    def constraints_overlap(vars):
        # vars shape (78,)
        # Extract centers and radii
        c = vars.reshape(n_circles, 3) # Each row: [x, y, r]
        cons = []
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                ci = c[i, :2]
                cj = c[j, :2]
                ri = c[i, 2]
                rj = c[j, 2]
                dist = np.sqrt(np.sum((ci - cj)**2))
                # Constraint: dist >= ri + rj  =>  dist - ri - rj >= 0
                cons.append(dist - ri - rj)
        return cons

    def constraints_boundary(vars):
        # vars shape (78,)
        c = vars.reshape(n_circles, 3)
        cons = []
        for i in range(n_circles):
            x, y, r = c[i]
            # x >= r  => x - r >= 0
            cons.append(x - r)
            # x <= 1 - r => 1 - x - r >= 0
            cons.append(1 - x - r)
            # y >= r
            cons.append(y - r)
            # y <= 1 - r
            cons.append(1 - y - r)
            # r >= 0 (handled by bounds usually, but can add constraint)
            cons.append(r) 
        return cons

    # Flatten initial guess
    init_guess = np.zeros(n_circles * 3)
    for i in range(n_circles):
        init_guess[3*i] = centers_init[i, 0]
        init_guess[3*i + 1] = centers_init[i, 1]
        init_guess[3*i + 2] = radii_init[i]

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius is 0.5)
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r

    # Constraints in SLSQP format: {'type': 'ineq', 'fun': fun}
    # ineq means fun(x) >= 0
    
    # Overlap constraints (many)
    # Creating a list of constraint dicts might be heavy, but 325 is manageable.
    overlap_constraints = []
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            def make_overlap_con(i, j):
                def con(vars):
                    ci = vars[3*i:3*i+2]
                    cj = vars[3*j:3*j+2]
                    ri = vars[3*i+2]
                    rj = vars[3*j+2]
                    dist = np.sqrt(np.sum((ci - cj)**2))
                    return dist - ri - rj
                return con
            overlap_constraints.append({'type': 'ineq', 'fun': make_overlap_con(i, j)})

    # Boundary constraints
    boundary_constraints = []
    for i in range(n_circles):
        # x - r >= 0
        def make_wall_con(i, axis, sign):
            # axis 0 for x, 1 for y. sign +1 for lower (x-r), -1 for upper (1-x-r)
            def con(vars):
                coord = vars[3*i + axis]
                r = vars[3*i + 2]
                if sign == 1:
                    return coord - r
                else:
                    return 1 - coord - r
            return con
        boundary_constraints.append({'type': 'ineq', 'fun': make_wall_con(i, 0, 1)}) # x >= r
        boundary_constraints.append({'type': 'ineq', 'fun': make_wall_con(i, 0, -1)}) # x <= 1-r
        boundary_constraints.append({'type': 'ineq', 'fun': make_wall_con(i, 1, 1)}) # y >= r
        boundary_constraints.append({'type': 'ineq', 'fun': make_wall_con(i, 1, -1)}) # y <= 1-r
        
    # Combine constraints
    all_constraints = overlap_constraints + boundary_constraints

    # Run optimization
    # SLSQP is suitable for constrained optimization with bounds
    try:
        res = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=all_constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
        
        final_vars = res.x
        final_centers = np.array([[final_vars[3*i], final_vars[3*i+1]] for i in range(n_circles)])
        final_radii = np.array([final_vars[3*i+2] for i in range(n_circles)])
        
    except Exception as e:
        # Fallback to initial guess if optimization fails
        print(f"Optimization failed: {e}")
        final_centers = centers_init
        final_radii = radii_init

    sum_radii = np.sum(final_radii)
    
    # Ensure no negative radii due to numerical errors
    final_radii = np.maximum(final_radii, 0.0)
    
    return final_centers, final_radii, sum_radii

if __name__ == "__main__":
    centers, radii, s_r = run_packing()
    print(f"Sum of radii: {s_r}")
    # print(f"Centers:\n{centers}")
    # print(f"Radii:\n{radii}")
