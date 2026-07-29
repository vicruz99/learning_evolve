# sol_000148 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 46a34d55) state=9aedec79 sum of radii=2.611085 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    
    # --- 1. Initial Guess: Hexagonal Packing ---
    # We arrange circles in a hexagonal grid pattern.
    # Estimate radius based on packing density. 
    # For N=26, a hexagonal packing allows r ~ 0.1.
    r_init = 0.105
    
    # We need to place 26 circles. Let's try a distribution of rows.
    # Hexagonal packing rows alternate in length to fit in gaps.
    # Target distribution for 26 circles: 5, 5, 5, 5, 4, 2? 
    # Or maybe 5, 5, 5, 5, 5, 1? 
    # Let's try to fit a rectangular block with some extras.
    # 5 rows of 5 circles = 25. Add 1.
    # But 5 circles in a row need width 10r. If r=0.105, width=1.05 > 1.
    # So we can't have a full row of 5 with r=0.105.
    # We need to be careful with the initial guess to be valid.
    
    # Let's construct a valid hexagonal grid.
    # Row spacing dy = r * sqrt(3)
    # Col spacing dx = 2 * r
    # Offset for odd rows dx/2 = r
    
    # Let's try 6 rows.
    # Rows lengths: 4, 5, 4, 5, 4, 4? Sum = 26.
    # Max width needed for row of 5: 10r. If r=0.1, width=1.0. Fits exactly.
    # So let's start with r=0.09 to be safe and let optimizer grow it.
    
    r_start = 0.09
    centers_init = []
    
    # Rows configuration: lengths
    # Try to pack tightly. 
    # Row 0: 5 circles
    # Row 1: 4 circles (offset)
    # Row 2: 5 circles
    # Row 3: 4 circles
    # Row 4: 5 circles
    # Row 5: 3 circles? Sum = 26.
    # 5+4+5+4+5+3 = 26.
    
    row_lengths = [5, 4, 5, 4, 5, 3]
    
    dy = r_start * np.sqrt(3)
    dx = 2 * r_start
    
    # Center vertically
    total_height = (len(row_lengths) - 1) * dy
    y_start = r_start + (1 - 2 * r_start - total_height) / 2
    
    # Center horizontally based on max width
    # Max width for 5 circles is 10*r. 
    # We want to fit in [r, 1-r].
    # Let's just place them and shift later if needed, or place centered.
    # Actually, let's place them such that they are centered in the square initially.
    
    # However, simpler: place them in a grid starting from (r, r)
    # and let the optimizer move them.
    # To avoid boundary issues initially, let's place them with some margin.
    
    y = r_start + 0.1 # Start with some margin
    for i, count in enumerate(row_lengths):
        # Offset for this row
        if i % 2 == 0:
            x_start = r_start
        else:
            x_start = r_start + r_start # Shift by r (half dx)
            
        for j in range(count):
            x = x_start + j * dx
            centers_init.append([x, y])
        y += dy
        
    centers_init = np.array(centers_init)
    radii_init = np.full(n, r_start)
    
    # Check if initial guess is valid (optional, but good for debugging)
    # If not valid, the optimizer might struggle, but SLSQP can handle infeasible starts usually.
    # With r=0.09, 5 circles width = 0.9. Fits in 1.0.
    # Height: 6 rows. dy = 0.09 * 1.732 = 0.156.
    # Total height span ~ 5 * 0.156 = 0.78. Fits in 1.0.
    # So initial guess is likely valid.
    
    # --- 2. Optimization ---
    
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # Total 3 * n variables.
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    def objective(vars):
        # Maximize sum of radii -> Minimize negative sum
        return -sum(vars[3*i+2] for i in range(n))
        
    def constraint_boundary(vars):
        # x >= r, 1-x >= r => r <= x <= 1-r
        # y >= r, 1-y >= r => r <= y <= 1-r
        cons = []
        for i in range(n):
            x = vars[3*i]
            y = vars[3*i+1]
            r = vars[3*i+2]
            cons.append(x - r)       # x >= r
            cons.append(1 - x - r)   # 1-x >= r
            cons.append(y - r)       # y >= r
            cons.append(1 - y - r)   # 1-y >= r
        return cons
        
    def constraint_overlap(vars):
        # dist >= r_i + r_j
        # sqrt(dx^2 + dy^2) >= r_i + r_j
        # To avoid sqrt in constraint (non-smooth at 0), we can square it?
        # But r_i+r_j can be 0. 
        # Actually, standard form g(vars) >= 0.
        # g = dist - (r_i + r_j)
        cons = []
        for i in range(n):
            xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
            for j in range(i + 1, n):
                xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                cons.append(dist - (ri + rj))
        return cons

    # Prepare constraints for scipy
    # scipy expects a list of NonlinearConstraint or dict
    # Using dict style for minimize
    
    # We can combine boundary constraints
    cons_boundary = {'type': 'ineq', 'fun': constraint_boundary}
    cons_overlap = {'type': 'ineq', 'fun': constraint_overlap}
    
    constraints = [cons_boundary, cons_overlap]
    
    # Bounds for variables
    # x, y in [0, 1], r >= 0
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (cannot be larger than 0.5)
        
    # Run optimizer
    # method='SLSQP' is suitable
    # tol=1e-9 for high precision
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
    )
    
    # --- 3. Extract Results ---
    if result.success or result.fun > -3.0: # Just a sanity check
        centers_opt = np.zeros((n, 2))
        radii_opt = np.zeros(n)
        for i in range(n):
            centers_opt[i, 0] = result.x[3*i]
            centers_opt[i, 1] = result.x[3*i+1]
            radii_opt[i] = result.x[3*i+2]
    else:
        # Fallback to initial if optimization fails completely
        centers_opt = centers_init
        radii_opt = radii_init

    # Clean up any tiny negative radii due to numerical error
    radii_opt = np.maximum(radii_opt, 0.0)
    
    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii

# Validation check (internal)
if __name__ == "__main__":
    centers, radii, s = run_packing()
    # Print some stats
    print(f"Sum of radii: {s}")
    print(f"Max radius: {np.max(radii)}, Min radius: {np.min(radii)}")
