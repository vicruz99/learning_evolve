# sol_000188 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4dd6d242) state=3cc9a5a2 sum of radii=2.613222 correctness=1.0
# stdout(first 200): Optimization terminated successfully    (Exit mode 0)             Current function value: -2.6132223728808954             Iterations: 11             Function evaluations: 790             Gradient eval
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

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimize the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    # Variables: [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    # Total variables = 26 * 3 = 78
    
    # Initialize with a hexagonal grid
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Parameters for hexagonal packing
    cols = 6
    rows = 5
    spacing_x = 0.2
    spacing_y = 0.15
    radius_init = 0.04
    
    # Generate initial positions
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= n:
                break
            # Hexagonal shift
            x = (c + 0.5) * spacing_x
            y = (r + 0.5) * spacing_y + (c % 2) * (spacing_y / 2)
            
            # Adjust to fit in unit square [0, 1]
            x = 0.1 + x * 0.8
            y = 0.1 + y * 0.8
            
            if count < n:
                centers[count] = [x, y]
                radii[count] = radius_init
                count += 1
        if count >= n:
            break
            
    # Flatten for optimizer
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds for variables
    bounds = []
    for i in range(n):
        # x, y in [0, 1], r >= 0
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (upper bound 0.5 is safe)
        
    # Objective function: maximize sum of radii => minimize negative sum
    def objective(vars):
        r = vars[2::3]
        return -np.sum(r)
        
    # Constraints
    constraints = []
    
    # Boundary constraints: r <= x <= 1-r  =>  x >= r, 1-x >= r
    # r <= y <= 1-y  =>  y >= r, 1-y >= r
    for i in range(n):
        idx = 3 * i
        xi = lambda v, i=i: v[idx]
        yi = lambda v, i=i: v[idx+1]
        ri = lambda v, i=i: v[idx+2]
        
        # x >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - x >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
        # y >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - y >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})
        
    # Non-overlap constraints: ||ci - cj|| >= ri + rj
    # ||ci - cj||^2 >= (ri + rj)^2
    # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = 3 * i
            idx_j = 3 * j
            
            def make_constraint(i, j):
                def constraint_func(v):
                    xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
                    xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    rad_sum_sq = (ri + rj)**2
                    return dist_sq - rad_sum_sq
                return constraint_func
            
            constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})
            
    # Run optimization
    # SLSQP is a good choice for non-linear constraints
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 1000, 'ftol': 1e-9, 'disp': True})
    
    if res.success:
        final_centers = np.zeros((n, 2))
        final_radii = np.zeros(n)
        for i in range(n):
            final_centers[i, 0] = res.x[3*i]
            final_centers[i, 1] = res.x[3*i+1]
            final_radii[i] = res.x[3*i+2]
            
        # Ensure radii are not negative due to numerical errors
        final_radii = np.maximum(final_radii, 0.0)
        
        # Validate
        if validate_packing(final_centers, final_radii):
            sum_radii = np.sum(final_radii)
            return final_centers, final_radii, sum_radii
        else:
            print("Optimization succeeded but validation failed. Falling back to initial.")
            return centers, radii, np.sum(radii)
    else:
        print("Optimization failed.")
        return centers, radii, np.sum(radii)

# Execute the packing
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print(f"Number of circles: {len(radii)}")
    print(f"Min radius: {np.min(radii)}, Max radius: {np.max(radii)}")
