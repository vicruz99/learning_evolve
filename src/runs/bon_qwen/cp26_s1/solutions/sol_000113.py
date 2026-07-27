# sol_000113 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state eb34cb51) state=1adc9047 sum of radii=2.611939 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def run_packing():
    n = 26
    
    # --- Initialization ---
    # We try to place circles in a hexagonal grid pattern.
    # Estimate radius around 0.1.
    r_init = 0.09 # Start slightly smaller to ensure feasibility
    
    # Generate hexagonal grid points
    # Vertical spacing: r * sqrt(3)
    # Horizontal spacing: 2 * r
    # Shift odd rows by r
    
    pts = []
    y = r_init
    row = 0
    # We want to cover the square. 
    # Height constraint: y + r <= 1 => y <= 1 - r
    while y <= 1 - r_init + 0.01: # +epsilon to catch last row
        # Determine x start for this row
        if row % 2 == 0:
            x_start = r_init
        else:
            x_start = 2 * r_init # Shifted by r (since spacing is 2r, shift r puts it in middle)
            # Actually standard hex: row 0 centers at r, 3r... row 1 centers at 2r, 4r...
            # Distance between (r, 0) and (2r, sqrt(3)r) is sqrt(r^2 + 3r^2) = 2r. Correct.
        
        x = x_start
        while x <= 1 - r_init + 0.01:
            pts.append([x, y])
            x += 2 * r_init
        
        y += r_init * np.sqrt(3)
        row += 1
    
    # We need exactly 26 points. 
    # If we have more, trim. If fewer, pad or adjust.
    # With r=0.09, we likely have plenty.
    if len(pts) < n:
        # Fallback to random if grid is too sparse (unlikely)
        pts = [np.random.rand(2) * 0.8 + 0.1 for _ in range(n)]
    
    # Select 26 points. Heuristic: pick points that are most "central" or just first 26?
    # First 26 in scan order is fine.
    initial_centers = np.array(pts[:n])
    initial_radii = np.full(n, 0.01) # Start with small radii to avoid constraint violation at start
    
    # Flatten variables: [x0, y0, ..., x25, y25, r0, ..., r25]
    x0 = np.concatenate([initial_centers.flatten(), initial_radii])
    
    # --- Optimization Setup ---
    
    def objective(vars):
        # Maximize sum of radii => Minimize negative sum
        radii = vars[52:]
        return -np.sum(radii)

    constraints = []

    # Non-overlap constraints: dist^2 >= (ri + rj)^2
    # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            def inequality(i=i, j=j):
                def fun(vars):
                    xi, yi = vars[2*i], vars[2*i+1]
                    xj, yj = vars[2*j], vars[2*j+1]
                    ri, rj = vars[52+i], vars[52+j]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    return dist_sq - (ri + rj)**2
                return fun
            constraints.append({'type': 'ineq', 'fun': inequality()})

    # Boundary constraints
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(n):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[52+i]})
        # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i] - v[52+i]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[52+i]})
        # 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i+1] - v[52+i]})
        
        # r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[52+i]})

    # Bounds for variables (helps solver)
    # x, y in [0, 1]
    # r in [0, 0.5] (cannot be larger than 0.5 in unit square)
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
    for _ in range(n):
        bounds.append((0, 0.5)) # r

    # Run optimization
    # SLSQP is suitable for constrained non-linear optimization
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                      options={'maxiter': 1000, 'ftol': 1e-9})
    
    if result.success:
        final_vars = result.x
    else:
        # Fallback to result if not successful (might still be valid)
        final_vars = result.x
        
    centers = final_vars[:52].reshape(n, 2)
    radii = final_vars[52:]
    
    # Ensure radii are non-negative (clamping)
    radii = np.maximum(radii, 0)
    
    # Update centers based on radii to ensure boundary validity strictly if needed
    # (The solver should have handled it, but numerical noise might slip)
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)
        
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# To test locally (optional logic inside run_packing is sufficient for the prompt)
# if __name__ == "__main__":
#     c, r, s = run_packing()
#     print(validate_packing(c, r))
#     print(f"Sum: {s}")
