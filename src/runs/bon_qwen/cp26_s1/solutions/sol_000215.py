# sol_000215 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0926bf18) state=9caa671f sum of radii=2.519801 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # 1. Initialize centers using a hexagonal lattice pattern
    # We aim to pack points as densely as possible. 
    # A 5x5 grid would be 25 circles. We need 26.
    # A hexagonal arrangement allows denser packing.
    # Let's try a rough estimate of radius ~0.1 to place them.
    r_init = 0.1
    
    # Create a hexagonal grid of points
    # Spacing x = 2*r, y = r*sqrt(3)
    # We want to fill the [0,1]x[0,1] square.
    # Let's generate more points than needed and pick the best 26?
    # Or just construct a specific layout.
    # Let's try to fit rows. 
    # With r=0.1, diameter 0.2.
    # Rows at y = r, r + r*sqrt(3), ...
    # y positions: 0.1, 0.273, 0.446, 0.619, 0.793, 0.966
    # 6 rows possible.
    # In each row, x spacing 2*r = 0.2.
    # x positions: 0.1, 0.3, 0.5, 0.7, 0.9
    # 5 columns.
    # 6 rows * 5 cols = 30 points. We can select 26.
    
    centers = []
    r_est = 0.09 # Slightly smaller to ensure fit initially
    
    # Generate hexagonal grid points
    # Offset every other row
    rows = []
    y = r_est
    while y <= 1 - r_est + 1e-9:
        row = []
        x = r_est
        # Check if row is even or odd index to shift
        # Actually, let's just generate a dense grid and pick best?
        # Or construct specific rows.
        # Row 0: 5 circles
        # Row 1: 5 circles (shifted by r_est)
        # ...
        # If shifted, x starts at r_est + r_est? No, x starts at r_est.
        # If shifted by r_est, x coordinates are x0 + r_est.
        # Let's just generate a standard grid and perturb?
        # Better: explicit construction.
        
        # Determine if this row is shifted
        row_idx = len(rows)
        shift = r_est if row_idx % 2 == 1 else 0
        
        x_curr = r_est + shift
        # Max x is 1 - r_est
        while x_curr <= 1 - r_est + 1e-9:
            centers.append([x_curr, y])
            x_curr += 2 * r_est
        rows.append(y)
        y += r_est * np.sqrt(3)
        
    # We might have too many or too few. 
    # With r=0.09, 6 rows likely.
    # Let's just take the first 26 generated or distribute them.
    # Actually, simply taking the generated centers might leave gaps or have overlaps if r is tight.
    # But for initialization, it's fine.
    
    # Ensure we have exactly 26
    if len(centers) > n:
        centers = centers[:n]
    elif len(centers) < n:
        # If not enough, add random points or fill gaps
        # This is unlikely with r=0.09
        pass
        
    centers = np.array(centers)
    
    # 2. Define optimization variables and functions
    # Variables: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Initial values
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = 0.1 # Initial radius guess
        
    # Objective: Maximize sum of radii => Minimize negative sum
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)

    # Constraints
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # This can be written as:
    # x - r >= 0
    # 1 - r - x >= 0
    # y - r >= 0
    # 1 - r - y >= 0
    
    # Non-overlap constraints: dist(i, j) >= r_i + r_j
    # sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
    
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        # x >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[3*idx] - v[3*idx+2]
        })
        # 1 - x >= r => 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[3*idx] - v[3*idx+2]
        })
        # y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: v[3*idx+1] - v[3*idx+2]
        })
        # 1 - y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, idx=i: 1.0 - v[3*idx+1] - v[3*idx+2]
        })
        
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            def dist_constraint(v, i=i, j=j):
                xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                return dist - (ri + rj)
            
            constraints.append({
                'type': 'ineq',
                'fun': dist_constraint
            })

    # 3. Optimization
    # Use SLSQP method
    # Bounds for variables: x, y in [0, 1], r >= 0
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (cannot exceed 0.5)
        
    # To avoid local minima, we can run multiple times or rely on good init.
    # Given the complexity, a single run with good init is best effort.
    # We might need to adjust the initial r to ensure feasibility.
    # If init is infeasible, optimizer might struggle.
    # Let's scale down initial radii to ensure valid start.
    
    # Re-initialize with small valid radii
    x0_opt = np.zeros(3 * n)
    # Place centers on a grid first to ensure valid start
    grid_step = 1.0 / 6.0
    idx = 0
    for r in range(6):
        for c in range(6):
            if idx < n:
                cx = (c + 0.5) * grid_step
                cy = (r + 0.5) * grid_step
                x0_opt[3*idx] = cx
                x0_opt[3*idx+1] = cy
                x0_opt[3*idx+2] = 0.05 # Small radius
                idx += 1
    if idx < n:
        # Fill remaining if any (should be 36 spots for 6x6)
        pass

    # Try to optimize
    try:
        res = minimize(objective, x0_opt, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'ftol': 1e-12, 'maxiter': 1000, 'disp': False})
        
        if res.success or res.fun < -2.0: # If we found something decent
            optimal_vars = res.x
        else:
            # Fallback to heuristic if optimization fails
            optimal_vars = x0_opt
    except Exception:
        optimal_vars = x0_opt

    # Extract results
    centers_opt = np.zeros((n, 2))
    radii_opt = np.zeros(n)
    
    for i in range(n):
        centers_opt[i, 0] = optimal_vars[3*i]
        centers_opt[i, 1] = optimal_vars[3*i+1]
        radii_opt[i] = optimal_vars[3*i+2]
        
    # Clean up numerical noise (ensure radii non-negative and valid)
    radii_opt = np.maximum(radii_opt, 0.0)
    
    # Final validation and adjustment (simple projection)
    # If any circle is outside, clamp it
    for i in range(n):
        r = radii_opt[i]
        x, y = centers_opt[i]
        # Clamp center to be at least r away from boundary
        x = max(r, min(1-r, x))
        y = max(r, min(1-r, y))
        centers_opt[i] = [x, y]
        
    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii

# For local testing, you can call run_packing() and check validate_packing
if __name__ == "__main__":
    # Mock validation import if needed, but here we just run
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic check
    # from validate_packing import validate_packing
    # print(validate_packing(c, r))
