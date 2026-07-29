# sol_000222 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96713eb2) state=7743ac28 sum of radii=2.510000 correctness=1.0
# stdout(first 200): Circle 0 at (0.09, 0.09) with radius 0.5 is outside the unit square Circle 0 at (0.1, 0.1) with radius 0.5 is outside the unit square Circle 0 at (0.08316800184590394, 0.1058760134481089) with radius 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import itertools

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

def get_initial_guess_hexagonal(n_circles):
    """Generates a hexagonal packing initial guess."""
    # Estimate radius for n_circles in a hexagonal packing roughly
    # Area ~ n * pi * r^2 <= 0.9 (density). r ~ sqrt(0.9 / (n * pi))
    # For 26, r ~ 0.106. Let's start with r=0.09 to be safe and let optimizer grow.
    r_guess = 0.09
    
    centers = []
    y = r_guess
    row_idx = 0
    while len(centers) < n_circles:
        x_start = r_guess if row_idx % 2 == 0 else 2 * r_guess
        x = x_start
        while x <= 1 - r_guess:
            if len(centers) < n_circles:
                centers.append([x, y])
            x += 2 * r_guess
        y += np.sqrt(3) * r_guess
        row_idx += 1
        
    # If we didn't get enough, just add random ones or extend
    while len(centers) < n_circles:
        centers.append([np.random.rand(), np.random.rand()])
        
    centers = np.array(centers[:n_circles])
    radii = np.full(n_circles, r_guess)
    return centers, radii

def get_initial_guess_grid_plus(n_circles):
    """Generates a 5x5 grid plus extra circles."""
    # 5x5 grid has 25 circles. We need 26.
    # Place 25 in a grid, 1 in the center of a hole.
    centers = []
    r_grid = 0.1
    # 5x5 grid centers
    for i in range(5):
        for j in range(5):
            x = 0.1 + i * 0.2
            y = 0.1 + j * 0.2
            centers.append([x, y])
    
    # Add 26th circle in a gap, e.g., at (0.2, 0.2) relative to grid?
    # Grid points are 0.1, 0.3, ...
    # Hole at (0.2, 0.2). Distance to (0.1, 0.1) is sqrt(0.02) ~ 0.141.
    # r_hole + r_grid <= 0.141. r_hole <= 0.041.
    centers.append([0.2, 0.2])
    
    centers = np.array(centers[:n_circles])
    radii = np.array([0.09] * 25 + [0.03]) # Start smaller to allow optimization
    return centers, radii

def objective(vars, n):
    # vars contains x1, y1, r1, x2, y2, r2, ...
    radii = vars[2::3]
    return -np.sum(radii)

def constraints_factory(n):
    cons = []
    
    # Boundary constraints: r <= x <= 1-r  => x - r >= 0, 1 - x - r >= 0
    # y - r >= 0, 1 - y - r >= 0
    # r >= 0
    
    for i in range(n):
        idx_x = 3*i
        idx_y = 3*i + 1
        idx_r = 3*i + 2
        
        # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_x] - v[idx_r]})
        # 1 - x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[idx_x] - v[idx_r]})
        # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_y] - v[idx_r]})
        # 1 - y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[idx_y] - v[idx_r]})
        # r >= 0 (handled by bounds, but good for safety)
        # cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[idx_r]})

    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi, idx_yi, idx_ri = 3*i, 3*i + 1, 3*i + 2
            idx_xj, idx_yj, idx_rj = 3*j, 3*j + 1, 3*j + 2
            
            def non_overlap(v, i=i, j=j):
                xi, yi, ri = v[idx_xi], v[idx_yi], v[idx_ri]
                xj, yj, rj = v[idx_xj], v[idx_yj], v[idx_rj]
                dist_sq = (xi - xj)**2 + (yi - yj)**2
                rad_sum_sq = (ri + rj)**2
                return dist_sq - rad_sum_sq
            
            cons.append({'type': 'ineq', 'fun': non_overlap})
            
    return cons

def run_packing() -> tuple:
    n = 26
    constraints = constraints_factory(n)
    bounds = [(0, 1) for _ in range(2*n)] + [(0, 0.5) for _ in range(n)]
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Strategy 1: Hexagonal Initialization
    centers_hex, radii_hex = get_initial_guess_hexagonal(n)
    x0_hex = np.zeros(3*n)
    for i in range(n):
        x0_hex[3*i] = centers_hex[i, 0]
        x0_hex[3*i+1] = centers_hex[i, 1]
        x0_hex[3*i+2] = radii_hex[i]
        
    # Strategy 2: Grid Initialization
    centers_grid, radii_grid = get_initial_guess_grid_plus(n)
    x0_grid = np.zeros(3*n)
    for i in range(n):
        x0_grid[3*i] = centers_grid[i, 0]
        x0_grid[3*i+1] = centers_grid[i, 1]
        x0_grid[3*i+2] = radii_grid[i]
        
    initial_guesses = [x0_hex, x0_grid]
    
    # Add some random perturbations to hex guess
    for _ in range(2):
        perturbed_centers = centers_hex + np.random.normal(0, 0.01, size=centers_hex.shape)
        perturbed_centers = np.clip(perturbed_centers, 0.05, 0.95) # Ensure inside
        x0_pert = np.zeros(3*n)
        for i in range(n):
            x0_pert[3*i] = perturbed_centers[i, 0]
            x0_pert[3*i+1] = perturbed_centers[i, 1]
            x0_pert[3*i+2] = radii_hex[i]
        initial_guesses.append(x0_pert)

    for k, x0 in enumerate(initial_guesses):
        try:
            res = minimize(
                objective, 
                x0, 
                args=(n,),
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < best_sum: # res.fun is negative sum
                current_sum = -res.fun
                # Extract solution
                c = res.x[:2*n].reshape((n, 2))
                r = res.x[2*n:]
                
                # Validate
                if validate_packing(c, r):
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = c
                        best_radii = r
                else:
                    # If invalid due to numerical issues, try to repair slightly
                    # Reduce radii slightly to satisfy constraints
                    # This is a fallback, ideally constraints are satisfied
                    pass
        except Exception as e:
            print(f"Optimization failed for guess {k}: {e}")

    # Fallback to a valid simple packing if optimization fails
    if best_centers is None:
        # Simple grid
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < n:
                    centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                    radii[idx] = 0.1
                    idx += 1
        # If we need more, add small ones? But n=26, grid has 25.
        # Just return grid with 25 circles and 1 tiny circle? 
        # But we need 26 circles.
        # Let's just use the hex result if available, otherwise this fallback.
        # The hex result should be valid.
        if idx < n:
             centers[idx] = [0.2, 0.2]
             radii[idx] = 0.01
             idx += 1
        
        best_centers = centers[:n]
        best_radii = radii[:n]
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum
