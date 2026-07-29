# sol_000279 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9d8cea89) state=743f5585 sum of radii=2.598726 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def get_initial_hexagonal(n_circles):
    """
    Generate an initial hexagonal packing configuration for n_circles.
    """
    centers = []
    radii = []
    
    # Try to form a hexagonal lattice
    # Estimate radius for initial guess
    # For 26 circles, approx radius 0.1
    r_init = 0.08 
    h = r_init * math.sqrt(3)
    
    # We will try to fit circles row by row
    # Start with a grid and shift every other row
    row_circles = []
    
    # Heuristic to distribute n_circles into rows
    # Aim for roughly 5 circles per row
    # 26 / 5 = 5.2 rows.
    # Let's try pattern: 5, 4, 5, 4, 5, 3
    
    row_lengths = [5, 4, 5, 4, 5, 3]
    count = 0
    current_rows = []
    for length in row_lengths:
        current_rows.append(length)
        count += length
        if count >= n_circles:
            # Trim the last row if needed
            excess = count - n_circles
            current_rows[-1] -= excess
            break
    
    # Adjust last row if necessary (should be exact if logic is correct)
    # Actually simpler: just fill rows until n_circles reached
    centers = []
    y = r_init
    
    # Shift pattern: 0, 0.5, 0, 0.5 ... (in units of 2r)
    # x spacing is 2r
    
    row_idx = 0
    total_packed = 0
    
    # Let's try to fit as many as possible with length 5
    # 5, 5, 5, 5, 4, 2?
    # Let's just use a simple loop
    
    while total_packed < n_circles:
        # Determine max circles in this row based on remaining
        remaining = n_circles - total_packed
        if row_idx % 2 == 0:
            # Even row: try 5 circles
            num_in_row = min(5, remaining)
            shift = 0
        else:
            # Odd row: try 4 circles (shifted)
            num_in_row = min(4, remaining)
            shift = r_init # Shift by r (half of 2r)
            
        if num_in_row == 0:
            # Fallback if min logic fails or row_idx parity issues
            num_in_row = 1
            shift = 0

        # Place circles
        width_available = 1.0
        # Center the row
        total_width = num_in_row * 2 * r_init
        x_start = (1.0 - total_width) / 2.0 + shift
        
        for k in range(num_in_row):
            x = x_start + k * 2 * r_init
            # Adjust x if shifted to stay within bounds? 
            # With shift r_init, x_start might be negative?
            # Let's clamp or adjust.
            # Actually, simpler: just place them. Optimization will fix.
            centers.append([x, y])
            radii.append(r_init)
            total_packed += 1
        
        if total_packed >= n_circles:
            break
            
        y += h
        row_idx += 1

    return np.array(centers), np.array(radii)

def objective(vars, n):
    """
    Objective: maximize sum of radii -> minimize negative sum
    """
    radii = vars[2*n:]
    return -np.sum(radii)

def constraint_boundary(vars, n):
    """
    Constraints:
    r <= x <= 1-r
    r <= y <= 1-r
    r >= 0
    """
    x = vars[0:n]
    y = vars[n:2*n]
    r = vars[2*n:]
    
    constraints = []
    for i in range(n):
        # x >= r  => x - r >= 0
        constraints.append(x[i] - r[i])
        # x <= 1-r => 1 - x - r >= 0
        constraints.append(1.0 - x[i] - r[i])
        # y >= r
        constraints.append(y[i] - r[i])
        # y <= 1-r
        constraints.append(1.0 - y[i] - r[i])
        # r >= 0 (handled by bounds usually, but let's add slack if needed, 
        # but SLSQP handles bounds well)
        
    return np.array(constraints)

def constraint_overlap(vars, n):
    """
    Constraint: dist >= r_i + r_j
    (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    """
    x = vars[0:n]
    y = vars[n:2*n]
    r = vars[2*n:]
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dr = r[i] + r[j]
            dist_sq = dx*dx + dy*dy
            constraints.append(dist_sq - dr*dr)
            
    return np.array(constraints)

def run_packing():
    n = 26
    
    # We will try a few random initializations and pick the best
    best_score = -np.inf
    best_vars = None
    best_centers = None
    best_radii = None
    
    # Define bounds
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius)
    lb = np.zeros(3 * n)
    ub = np.ones(3 * n)
    ub[2*n:] = 0.5 # Radii max 0.5
    bounds = list(zip(lb, ub))

    # Define constraints
    # SLSQP supports nonlinear constraints
    cons = [
        {'type': 'ineq', 'fun': lambda v: constraint_boundary(v, n)},
        {'type': 'ineq', 'fun': lambda v: constraint_overlap(v, n)}
    ]
    
    # Initializations
    initial_guesses = []
    
    # 1. Hexagonal packing guess
    c, r = get_initial_hexagonal(n)
    # Flatten
    x = c[:, 0]
    y = c[:, 1]
    v0_hex = np.concatenate([x, y, r])
    initial_guesses.append(v0_hex)
    
    # 2. Random perturbation of grid
    np.random.seed(42)
    # 5x5 grid centers
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    cx, cy = np.meshgrid(grid_x, grid_y)
    centers_grid = np.vstack([cx.ravel(), cy.ravel()]).T # 25 points
    # Add one point in center
    centers_grid = np.vstack([centers_grid, [0.5, 0.5]])
    # Add small random noise
    noise = np.random.uniform(-0.01, 0.01, (n, 2))
    centers_grid += noise
    # Clip
    centers_grid = np.clip(centers_grid, 0.01, 0.99)
    radii_grid = np.full(n, 0.08)
    
    x_grid = centers_grid[:, 0]
    y_grid = centers_grid[:, 1]
    v0_grid = np.concatenate([x_grid, y_grid, radii_grid])
    initial_guesses.append(v0_grid)
    
    # 3. Another random initialization
    centers_rand = np.random.uniform(0.1, 0.9, (n, 2))
    radii_rand = np.full(n, 0.05)
    x_rand = centers_rand[:, 0]
    y_rand = centers_rand[:, 1]
    v0_rand = np.concatenate([x_rand, y_rand, radii_rand])
    initial_guesses.append(v0_rand)

    for i, v0 in enumerate(initial_guesses):
        try:
            # Maximize sum of radii => Minimize negative sum
            res = minimize(
                objective,
                v0,
                args=(n,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if res.success or (not res.success and -res.fun > best_score):
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_vars = res.x
                    # Extract centers and radii
                    best_centers = np.column_stack((best_vars[0:n], best_vars[n:2*n]))
                    best_radii = best_vars[2*n:]
        except Exception as e:
            print(f"Optimization failed for guess {i}: {e}")
            continue

    if best_centers is None:
        # Fallback to hexagonal if optimization failed
        best_centers, best_radii = get_initial_hexagonal(n)
        best_score = np.sum(best_radii)

    # Post-processing: Ensure validity and maybe fix minor numerical issues
    # The validation function is strict, but our constraints should handle it.
    # However, SLSQP might violate constraints slightly.
    # We can try to shrink radii slightly if validation fails, but let's hope it's good.
    
    # Validate
    if not validate_packing(best_centers, best_radii):
        print("Warning: Packing validation failed. Attempting to shrink radii.")
        # Simple heuristic shrink
        factor = 0.999
        while not validate_packing(best_centers, best_radii * factor) and factor > 0.9:
            factor -= 0.001
        best_radii = best_radii * factor
        best_score = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_score)

if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print(f"Validation: {validate_packing(centers, radii)}")
