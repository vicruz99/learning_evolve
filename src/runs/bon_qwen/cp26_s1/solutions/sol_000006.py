# sol_000006 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e6663bde) state=30c0f473 sum of radii=1.300000 correctness=1.0
# stdout(first 200): Optimization failed: 'function' object is not iterable Refinement failed: 'function' object is not iterable Sum of radii: 1.3000000000000003 Optimization failed: 'function' object is not iterable Refi
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

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
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.ndarray of shape (26, 2)
        radii: np.ndarray of shape (26,)
        sum_radii: float
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal Lattice Pattern
    # We create a grid of points that is dense and roughly hexagonal.
    # 26 circles: 5 rows of 5 and 1 row of 1? 
    # A pattern of 5, 4, 5, 4, 5, 4 would be 27. Let's use 5, 4, 5, 4, 5, 3 or similar.
    # Or simply a 6x6 grid subset.
    
    initial_centers = np.zeros((n_circles, 2))
    
    # Let's try to fit them in a slightly distorted hexagonal packing
    # Rows of 5 and 4
    # 5 + 4 + 5 + 4 + 5 + 3 = 26
    
    row_counts = [5, 4, 5, 4, 5, 3]
    y_spacing = 1.0 / (len(row_counts) + 1) # Approximate vertical spacing
    idx = 0
    
    for i, count in enumerate(row_counts):
        y = (i + 1) * y_spacing
        # Horizontal spacing
        if count == 0: continue
        x_spacing = 1.0 / (count + 1)
        for j in range(count):
            # Shift even rows by half spacing to create hexagonal effect
            offset = x_spacing / 2 if i % 2 == 1 else 0
            x = (j + 1) * x_spacing + offset
            initial_centers[idx] = [x, y]
            idx += 1
            
    # Random shuffle to break symmetry and avoid local minima traps
    np.random.seed(42)
    permutation = np.random.permutation(n_circles)
    initial_centers = initial_centers[permutation]

    # 2. Optimization Phase 1: Equal Radii
    # Maximize r such that all circles have radius r and fit.
    # Variables: x1, y1, ..., x26, y26, r
    # Total 53 variables.
    
    def objective_equal(params):
        # params: [x1, y1, ..., x26, y26, r]
        # We want to maximize r, so minimize -r
        r = params[-1]
        return -r

    def constraints_equal(params):
        centers = params[:2*n_circles].reshape(-1, 2)
        r = params[-1]
        cons = []
        
        # Boundary constraints
        for i in range(n_circles):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i] - p[-1]}) # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i] - p[-1]}) # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i+1] - p[-1]}) # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i+1] - p[-1]}) # 1 - y - r >= 0
            
        # Non-overlap constraints
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: 
                        np.sqrt((p[2*i] - p[2*j])**2 + (p[2*i+1] - p[2*j+1])**2) - 2 * p[-1]
                })
        return cons

    # Initial guess
    initial_r = 0.05
    initial_params = np.concatenate([initial_centers.flatten(), [initial_r]])

    # Optimize
    # Using SLSQP
    try:
        res1 = minimize(
            objective_equal, 
            initial_params, 
            method='SLSQP', 
            constraints=constraints_equal,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        
        if res1.success or (not np.isnan(res1.fun) and res1.fun > -0.5):
            optimal_params_equal = res1.x
            centers_opt = optimal_params_equal[:2*n_circles].reshape(-1, 2)
            r_opt = optimal_params_equal[-1]
            radii_opt = np.full(n_circles, r_opt)
        else:
            # Fallback if optimization fails
            centers_opt = initial_centers
            r_opt = 0.05
            radii_opt = np.full(n_circles, r_opt)
            
    except Exception as e:
        print(f"Optimization failed: {e}")
        centers_opt = initial_centers
        r_opt = 0.05
        radii_opt = np.full(n_circles, r_opt)

    # 3. Optimization Phase 2: Unequal Radii (Refinement)
    # Now allow radii to vary to squeeze out more sum.
    # Variables: x1, y1, ..., x26, y26, r1, ..., r26
    # Total 78 variables.
    
    def objective_unequal(params):
        radii = params[2*n_circles:]
        return -np.sum(radii)

    def constraints_unequal(params):
        centers = params[:2*n_circles].reshape(-1, 2)
        radii = params[2*n_circles:]
        cons = []
        
        # Boundary
        for i in range(n_circles):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i] - p[2*n_circles + i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i] - p[2*n_circles + i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i+1] - p[2*n_circles + i]})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[2*i+1] - p[2*n_circles + i]})
            
        # Non-overlap
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: 
                        np.sqrt((p[2*i] - p[2*j])**2 + (p[2*i+1] - p[2*j+1])**2) - (p[2*n_circles + i] + p[2*n_circles + j])
                })
        return cons

    initial_params_unequal = np.concatenate([centers_opt.flatten(), radii_opt])
    
    try:
        res2 = minimize(
            objective_unequal,
            initial_params_unequal,
            method='SLSQP',
            constraints=constraints_unequal,
            options={'ftol': 1e-10, 'maxiter': 2000}
        )
        
        if res2.success or (not np.isnan(res2.fun) and res2.fun > -2.0): # 2.0 is a reasonable lower bound
            final_params = res2.x
            final_centers = final_params[:2*n_circles].reshape(-1, 2)
            final_radii = final_params[2*n_circles:]
            
            # Clean up tiny radii or invalid values
            final_radii = np.maximum(final_radii, 1e-9)
        else:
            final_centers = centers_opt
            final_radii = radii_opt
            
    except Exception as e:
        print(f"Refinement failed: {e}")
        final_centers = centers_opt
        final_radii = radii_opt

    # 4. Final Validation and Adjustment
    # Sometimes optimizers push boundaries slightly. We enforce strict validity.
    
    # Clip centers to [0,1]
    final_centers = np.clip(final_centers, 0, 1)
    
    # Ensure circles are inside
    for i in range(n_circles):
        x, y = final_centers[i]
        r = final_radii[i]
        # If too close to boundary, reduce radius
        min_dist = min(x, 1-x, y, 1-y)
        if r > min_dist + 1e-12:
            final_radii[i] = max(0, min_dist - 1e-12)
            # Adjust center to be valid if radius became 0
            if final_radii[i] < 1e-9:
                final_radii[i] = 1e-9 # Keep a tiny circle
                # Center doesn't matter much for tiny circle, but ensure inside
                final_centers[i] = [0.5, 0.5]

    # Check overlaps and resolve by shrinking
    # This is a fallback for numerical instability
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.sqrt(np.sum((final_centers[i] - final_centers[j]) ** 2))
            req_dist = final_radii[i] + final_radii[j] + 1e-12
            if dist < req_dist:
                # Shrink the smaller circle or both
                overlap = req_dist - dist
                # Distribute shrinkage
                final_radii[i] -= overlap / 2
                final_radii[j] -= overlap / 2
                final_radii = np.maximum(final_radii, 0)

    sum_radii = np.sum(final_radii)
    
    # Verify
    if not validate_packing(final_centers, final_radii):
        print("Warning: Validation failed, returning fallback grid.")
        # Fallback to a safe grid
        fallback_centers = np.zeros((26, 2))
        idx = 0
        for r in range(5):
            for c in range(5):
                fallback_centers[idx] = [0.1 + c*0.2, 0.1 + r*0.2]
                idx += 1
        fallback_centers[25] = [0.5, 0.5]
        fallback_radii = np.full(26, 0.05)
        return fallback_centers, fallback_radii, 26*0.05

    return final_centers, final_radii, float(sum_radii)

# Run the packing to get the result
centers, radii, total_sum = run_packing()
print(f"Sum of radii: {total_sum}")
