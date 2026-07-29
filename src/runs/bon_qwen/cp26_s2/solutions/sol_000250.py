# sol_000250 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5bb01f44) state=690a33c1 sum of radii=2.588598 correctness=1.0
# stdout(first 200): Circle 0 at (0.07029329387955129, 0.07029329387970358) with radius 0.07029329388167752 is outside the unit square
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
    n = 26
    num_vars = n * 3  # x, y, r for each circle

    # --- Helper functions for constraints ---
    def boundary_constraints(vars):
        """Ensure circles are inside the unit square."""
        cons = []
        for i in range(n):
            x = vars[3 * i]
            y = vars[3 * i + 1]
            r = vars[3 * i + 2]
            # x - r >= 0
            cons.append(x - r)
            # 1 - (x + r) >= 0
            cons.append(1 - (x + r))
            # y - r >= 0
            cons.append(y - r)
            # 1 - (y + r) >= 0
            cons.append(1 - (y + r))
        return np.array(cons)

    def overlap_constraints(vars):
        """Ensure circles do not overlap."""
        cons = []
        for i in range(n):
            xi, yi, ri = vars[3 * i], vars[3 * i + 1], vars[3 * i + 2]
            for j in range(i + 1, n):
                xj, yj, rj = vars[3 * j], vars[3 * j + 1], vars[3 * j + 2]
                # dist^2 >= (ri + rj)^2
                # We use the squared distance to avoid sqrt and potential issues with small numbers,
                # though standard distance is fine too.
                dx = xi - xj
                dy = yi - yj
                dist_sq = dx*dx + dy*dy
                r_sum = ri + rj
                cons.append(dist_sq - r_sum**2)
        return np.array(cons)

    def radius_non_negativity(vars):
        """Ensure radii are non-negative."""
        radii = vars[2::3]
        return radii

    # --- Objective Function ---
    def objective(vars):
        """Maximize sum of radii -> Minimize negative sum."""
        radii = vars[2::3]
        return -np.sum(radii)

    # --- Initial Guess Generation ---
    # Generate a hexagonal packing arrangement
    # 26 circles: 5 rows with counts 6, 5, 5, 5, 5
    row_counts = [6, 5, 5, 5, 5]
    
    # Calculate initial centers based on a hexagonal lattice
    # We'll scale it to fit roughly in the unit square
    # Let's assume an initial radius r0 to calculate spacing
    # 2r0 * 6 <= 1  => r0 <= 1/12 approx 0.083
    # Height for 5 rows: 2r0 + 4*r0*sqrt(3) approx 2r0 + 6.9r0 = 8.9r0
    # 8.9 * 0.083 approx 0.74, fits in 1.
    
    r_est = 0.08
    initial_centers = []
    y_curr = r_est
    
    # We'll center the packing in the square
    # Calculate total width and height needed for the lattice first to center it
    
    # Approximate dimensions
    max_circles = max(row_counts)
    width_needed = 2 * r_est * max_circles
    height_needed = 2 * r_est + (len(row_counts) - 1) * r_est * np.sqrt(3)
    
    x_offset = (1.0 - width_needed) / 2.0
    y_offset = (1.0 - height_needed) / 2.0
    
    current_circle_idx = 0
    
    for row_idx, count in enumerate(row_counts):
        # Shift alternate rows by r_est (half spacing)
        # Standard hexagonal: spacing 2r, shift r
        if row_idx % 2 == 1:
            row_shift = r_est
        else:
            row_shift = 0
            
        for col_idx in range(count):
            # x coordinate: base + shift + col_spacing * col_idx
            # We want to center the row
            row_width = 2 * r_est * count
            x_start = x_offset + row_shift + (width_needed - row_width) / 2.0 # Center row within available width
            
            # Actually, simpler: just place them evenly
            # x = r_est + row_shift + col_idx * 2*r_est
            # Let's just generate them and then we can scale/shift if needed, 
            # but SLSQP should handle finding the position.
            # Let's just place them in a standard grid-like hex pattern starting from 0.
            
            # Let's try a more robust initial placement:
            # Scale the whole pattern to fit in [0,1]
            pass 
            
        # Let's rebuild the list properly
        pass

    # Re-building initial guess more carefully
    centers_list = []
    radii_list = []
    
    # Hexagonal grid parameters
    # dx = 2*r, dy = r*sqrt(3)
    # Let's pick a temporary r_temp = 0.1
    r_temp = 0.1
    dx = 2 * r_temp
    dy = r_temp * np.sqrt(3)
    
    # Create rows
    rows_data = []
    y_pos = r_temp
    for row_idx, count in enumerate(row_counts):
        # x positions
        # For even rows (0, 2, ...), start at r_temp
        # For odd rows (1, 3, ...), start at r_temp + dx/2 = 2r_temp? 
        # Wait, if spacing is dx=2r, shift is r.
        # So even: r, 3r, 5r...
        # odd: 2r, 4r, 6r...
        
        if row_idx % 2 == 0:
            x_start = r_temp
        else:
            x_start = 2 * r_temp
            
        row_centers = []
        for k in range(count):
            x = x_start + k * dx
            row_centers.append([x, y_pos])
        rows_data.append(row_centers)
        y_pos += dy
        
    # Collect all centers
    all_centers = []
    for row in rows_data:
        all_centers.extend(row)
        
    # Normalize/Scale to fit in [0,1]
    # Find bounds
    xs = [c[0] for c in all_centers]
    ys = [c[1] for c in all_centers]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Scale to fit in 0.9 x 0.9 to leave margin
    scale = 0.9 / max(width, height)
    
    final_centers = []
    for c in all_centers:
        x = (c[0] - min_x) * scale + (1.0 - 0.9)/2
        y = (c[1] - min_y) * scale + (1.0 - 0.9)/2
        final_centers.append([x, y])
        
    initial_radii = np.full(n, r_temp * scale)
    
    # Flatten into initial guess vector
    initial_vars = []
    for i in range(n):
        initial_vars.extend(final_centers[i])
        initial_vars.append(initial_radii[i])
    initial_vars = np.array(initial_vars)

    # --- Define Constraints for SLSQP ---
    # SLSQP expects constraints in the form dict with 'type' and 'fun'
    # Inequality constraints: fun(x) >= 0
    
    boundary_cons = {
        'type': 'ineq',
        'fun': boundary_constraints,
        'jac': None # SLSQP will approximate
    }
    
    overlap_cons = {
        'type': 'ineq',
        'fun': overlap_constraints,
        'jac': None
    }
    
    radius_cons = {
        'type': 'ineq',
        'fun': radius_non_negativity,
        'jac': None
    }

    constraints = [boundary_cons, overlap_cons, radius_cons]
    
    # --- Bounds ---
    # x, y in [0, 1]
    # r in [0, 0.5] (max possible radius)
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # --- Run Optimization ---
    try:
        # Use SLSQP
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
        )
        
        best_vars = result.x
        
    except Exception as e:
        # Fallback to initial guess if optimization fails
        best_vars = initial_vars
        print(f"Optimization failed: {e}")

    # --- Extract Results ---
    best_centers = np.array([[best_vars[3*i], best_vars[3*i+1]] for i in range(n)])
    best_radii = np.array([best_vars[3*i+2] for i in range(n)])
    
    # Clamp radii to ensure non-negative (numerical safety)
    best_radii = np.maximum(best_radii, 0.0)
    
    # Validate
    is_valid = validate_packing(best_centers, best_radii)
    if not is_valid:
        # If invalid, try to shrink radii slightly to fix overlaps?
        # Or just return what we have, but validation might fail.
        # A simple fix: reduce all radii by a small factor
        while not is_valid:
            best_radii *= 0.99
            is_valid = validate_packing(best_centers, best_radii)
            if np.sum(best_radii) < 1e-6: # Give up if radii too small
                break

    sum_radii = float(np.sum(best_radii))
    
    return best_centers, best_radii, sum_radii

# Example usage for testing
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    print(f"Valid: {validate_packing(centers, radii)}")
