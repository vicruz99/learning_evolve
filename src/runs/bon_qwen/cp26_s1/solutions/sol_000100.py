# sol_000100 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 15bab5cf) state=edd9bbbe sum of radii=2.530345 correctness=1.0
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

def objective(vars_flat, n_circles):
    # vars_flat contains x, y, r for each circle
    # We want to maximize sum of radii, so minimize negative sum
    radii = vars_flat[2 * n_circles:]
    return -np.sum(radii)

def constraint_boundary(vars_flat, n_circles):
    # Returns array of constraint values >= 0
    centers = vars_flat[:2 * n_circles].reshape(n_circles, 2)
    radii = vars_flat[2 * n_circles:]
    
    constraints = []
    for i in range(n_circles):
        x, y = centers[i]
        r = radii[i]
        # x - r >= 0
        constraints.append(x - r)
        # 1 - (x + r) >= 0
        constraints.append(1 - (x + r))
        # y - r >= 0
        constraints.append(y - r)
        # 1 - (y + r) >= 0
        constraints.append(1 - (y + r))
        # r >= 0
        constraints.append(r)
    return np.array(constraints)

def constraint_overlap(vars_flat, n_circles):
    # Returns array of constraint values >= 0
    # dist_ij - (ri + rj) >= 0
    centers = vars_flat[:2 * n_circles].reshape(n_circles, 2)
    radii = vars_flat[2 * n_circles:]
    
    constraints = []
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            constraints.append(dist - (radii[i] + radii[j]))
    return np.array(constraints)

def create_hexagonal_grid(n_circles):
    # Create a hexagonal grid arrangement for 26 circles
    # We will try to fit them in rows
    # Approximate rows: 5 rows. 
    # Distribution: 6, 5, 5, 5, 5 = 26
    # But 6 circles might be too wide. Let's try 5, 5, 5, 5, 6?
    # Or better, just place them in a dense hexagonal pattern and let optimizer fix it.
    
    centers = np.zeros((n_circles, 2))
    radii = np.full(n_circles, 0.08) # Initial guess radius
    
    # Simple hexagonal placement
    # Rows
    rows = []
    n_per_row = []
    # Try to balance rows
    # 26 circles. Sqrt(26) ~ 5.1.
    # 5 rows. 26/5 = 5.2.
    # Counts: 6, 5, 5, 5, 5
    
    # Let's try a more compact layout
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 6 circles? 
    # Actually, shifting allows fitting more.
    
    # Let's just generate a grid and shift every other row
    idx = 0
    y = 0.1
    row_idx = 0
    
    while idx < n_circles:
        # Determine how many circles in this row
        # Roughly 5 or 6
        if row_idx % 2 == 0:
            count = 6
        else:
            count = 5
            
        # Adjust count if we exceed n_circles
        if idx + count > n_circles:
            count = n_circles - idx
            
        # X positions
        if row_idx % 2 == 0:
            x_start = 0.1
        else:
            x_start = 0.1 + 0.08 # Shift by radius approx
            
        for k in range(count):
            if idx < n_circles:
                x = x_start + k * 0.16 # spacing approx 2*r
                if x < 1.0:
                    centers[idx] = [x, y]
                    idx += 1
        
        y += 0.14 # vertical spacing approx sqrt(3)*r
        row_idx += 1
        
    # Normalize radii to initial guess
    radii = np.full(n_circles, 0.09)
    
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n_circles = 26
    
    # We will run optimization multiple times from different starts
    best_sum_radii = -np.inf
    best_centers = None
    best_radii = None
    
    # Strategy 1: Hexagonal grid start
    centers_init, radii_init = create_hexagonal_grid(n_circles)
    
    # Strategy 2: Random start near grid
    centers_rand = np.random.uniform(0.1, 0.9, (n_circles, 2))
    radii_rand = np.full(n_circles, 0.08)
    
    starts = [
        np.concatenate([centers_init.flatten(), radii_init]),
        np.concatenate([centers_rand.flatten(), radii_rand])
    ]
    
    # Add a few more random starts
    for _ in range(3):
        c = np.random.uniform(0.1, 0.9, (n_circles, 2))
        r = np.full(n_circles, 0.09)
        starts.append(np.concatenate([c.flatten(), r]))

    for start_vars in starts:
        # Bounds for variables
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for _ in range(n_circles):
            bounds.append((0.0, 1.0)) # x
            bounds.append((0.0, 1.0)) # y
            bounds.append((0.0, 0.5)) # r
        
        # Constraints
        # Boundary constraints
        def cons_bound(x):
            return constraint_boundary(x, n_circles)
        
        # Overlap constraints
        def cons_overlap(x):
            return constraint_overlap(x, n_circles)

        # Using SLSQP
        # We need to define constraints properly for scipy
        # SLSQP supports 'ineq' constraints: g(x) >= 0
        
        cons = []
        
        # Boundary
        cons.append({
            'type': 'ineq',
            'fun': cons_bound
        })
        
        # Overlap
        cons.append({
            'type': 'ineq',
            'fun': cons_overlap
        })
        
        try:
            res = minimize(
                objective,
                start_vars,
                args=(n_circles,),
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < -2.5: # If we got a decent sum
                sum_r = -res.fun
                if sum_r > best_sum_radii:
                    best_sum_radii = sum_r
                    best_centers = res.x[:2 * n_circles].reshape(n_circles, 2)
                    best_radii = res.x[2 * n_circles:]
                    
        except Exception as e:
            pass

    # Final validation and cleanup
    if best_centers is not None:
        if validate_packing(best_centers, best_radii):
            return best_centers, best_radii, float(np.sum(best_radii))
    
    # Fallback if optimization failed or validation failed (e.g. numerical issues)
    # Return a valid simple packing
    centers_fallback, radii_fallback = create_hexagonal_grid(n_circles)
    # Ensure valid
    while not validate_packing(centers_fallback, radii_fallback):
        # Shrink radii slightly
        radii_fallback *= 0.99
    
    return centers_fallback, radii_fallback, float(np.sum(radii_fallback))

if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    print(centers)
    print(radii)
