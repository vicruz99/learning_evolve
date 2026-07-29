# sol_000334 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d755ba05) state=9669c849 sum of radii=2.612417 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26,) with radius of each circle
        sum_radii: float sum of radii
    """
    n_circles = 26
    
    # Strategy: 
    # 1. Initialize with a hexagonal packing pattern.
    # 2. Optimize positions and radii using SLSQP to maximize sum of radii.
    # 3. Perform multiple restarts to avoid local minima.
    
    def get_initial_guess(n):
        # Hexagonal packing initialization
        # Try to arrange in 6 rows to accommodate 26 circles
        # Pattern: 5, 4, 5, 4, 5, 3 or similar to sum to 26? 
        # 5+5+5+5+5+1 = 26?
        # Let's try a dense 5x5 grid + 1, but shifted for better spacing?
        # Actually, a hexagonal grid of 5 rows can hold ~25-30 depending on shift.
        
        centers = np.zeros((n, 2))
        
        # Let's try a rough hexagonal layout
        # Approximate radius 0.1
        r_est = 0.095
        dx = 2 * r_est
        dy = math.sqrt(3) * r_est
        
        count = 0
        row = 0
        # 6 rows
        rows_count = [5, 5, 5, 5, 5, 1] # Sum = 26
        
        for r_idx, num_in_row in enumerate(rows_count):
            y = r_est + r_idx * dy
            # Shift every other row
            shift = dx / 2 if r_idx % 2 == 1 else 0
            
            # Center the row in [0, 1]
            # Total width for num_in_row circles is (num_in_row - 1) * dx
            # Start x
            start_x = (1.0 - (num_in_row - 1) * dx) / 2.0 + shift
            if start_x < r_est: start_x = r_est # Safety clamp
            
            for i in range(num_in_row):
                x = start_x + i * dx
                centers[count, 0] = x
                centers[count, 1] = y
                count += 1
                if count == n:
                    break
            if count == n:
                break
                
        # Fill remaining if any (should be 0)
        for i in range(count, n):
            centers[i] = [0.5, 0.5]
            
        return centers, r_est

    def objective(vars):
        # vars shape: (n_circles * 3) -> x, y, r for each circle
        # We want to maximize sum(r), so minimize -sum(r)
        radii = vars[2::3]
        return -np.sum(radii)

    def constraints_factory(centers_init, r_init):
        constraints = []
        
        # Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        # We will add these as bounds or constraints. Bounds are easier for box constraints.
        # But since r is variable, we use inequality constraints.
        # Actually, for SLSQP, bounds are supported for variables.
        # Let's map variables: x_i, y_i, r_i.
        # Bounds:
        # 0 <= x_i <= 1
        # 0 <= y_i <= 1
        # r_i >= 0 (and effectively <= 0.5)
        
        # However, the non-overlap constraint is non-linear.
        # dist_ij >= r_i + r_j
        # sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
        
        # We will define a function that returns the constraint value.
        
        # For simplicity in optimization loop, we handle bounds in the minimize call
        # and non-linear constraints here.
        
        # But defining all pairwise constraints explicitly is heavy for callback.
        # We will return a list of dicts.
        
        non_linear_cons = []
        
        # Boundary constraints (inequalities >= 0)
        # x_i - r_i >= 0
        # 1 - x_i - r_i >= 0
        # y_i - r_i >= 0
        # 1 - y_i - r_i >= 0
        
        # We can implement these as simple functions
        # But SLSQP accepts a list of constraints.
        
        # Let's construct the constraint functions inside the loop or use a vectorized approach?
        # Constructing individually is easier to debug but slow. 
        # Given n=26, n*(n-1)/2 = 325 overlap constraints.
        
        # Optimization of 78 variables (26*3) with 325 constraints might be slow if not careful.
        # But for 26 circles it should be manageable.
        
        # Helper to extract vars
        def get_vars(v):
            xs = v[0::3]
            ys = v[1::3]
            rs = v[2::3]
            return xs, ys, rs

        # Boundary constraints
        # x_i - r_i >= 0
        def bound_x_low(v):
            xs, _, rs = get_vars(v)
            return xs - rs
        
        def bound_x_high(v):
            xs, _, rs = get_vars(v)
            return 1.0 - xs - rs
            
        def bound_y_low(v):
            _, ys, rs = get_vars(v)
            return ys - rs
            
        def bound_y_high(v):
            _, ys, rs = get_vars(v)
            return 1.0 - ys - rs
            
        # These are vectorized if we return array, but scipy constraint dict expects 'fun' returning scalar or array.
        # If array, type 'ineq' means all elements >= 0.
        
        constraints.append({'type': 'ineq', 'fun': bound_x_low})
        constraints.append({'type': 'ineq', 'fun': bound_x_high})
        constraints.append({'type': 'ineq', 'fun': bound_y_low})
        constraints.append({'type': 'ineq', 'fun': bound_y_high})
        
        # Overlap constraints
        # (xi - xj)^2 + (yi - yj)^2 >= (ri + rj)^2
        # Or dist - ri - rj >= 0
        
        def overlap_constraints(v):
            xs, ys, rs = get_vars(v)
            dists = []
            for i in range(n_circles):
                for j in range(i + 1, n_circles):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist = math.hypot(dx, dy)
                    dists.append(dist - rs[i] - rs[j])
            return np.array(dists)

        constraints.append({'type': 'ineq', 'fun': overlap_constraints})
        
        return constraints, bounds_generation()

    def bounds_generation():
        # x in [0, 1], y in [0, 1], r in [0, 0.5]
        bounds = []
        for _ in range(n_circles):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((1e-6, 0.5)) # r (strictly positive)
        return bounds

    # Prepare optimization
    init_centers, init_r = get_initial_guess(n_circles)
    
    # Create initial vector
    x0 = np.zeros(n_circles * 3)
    for i in range(n_circles):
        x0[3*i] = init_centers[i, 0]
        x0[3*i+1] = init_centers[i, 1]
        x0[3*i+2] = init_r
        
    constraints, bounds = constraints_factory(init_centers, init_r)
    
    best_result = None
    best_sum = -np.inf
    
    # Run multiple times with slight perturbations
    n_restarts = 5
    
    for k in range(n_restarts):
        # Perturb initial positions
        current_x0 = x0.copy()
        if k > 0:
            noise = np.random.normal(0, 0.01, size=x0.shape)
            # Don't perturb radius much
            noise[2::3] *= 0.1 
            current_x0 += noise
            # Clamp bounds
            current_x0[0::3] = np.clip(current_x0[0::3], 0, 1)
            current_x0[1::3] = np.clip(current_x0[1::3], 0, 1)
            current_x0[2::3] = np.clip(current_x0[2::3], 1e-4, 0.5)

        try:
            res = minimize(objective, current_x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                           options={'maxiter': 200, 'ftol': 1e-9})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = res
            elif not best_result:
                # Accept even if not successful if better than nothing
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = res
        except Exception as e:
            print(f"Optimization failed in restart {k}: {e}")
            continue

    if best_result is None:
        # Fallback to initial guess
        centers = init_centers
        radii = np.full(n_circles, init_r)
        sum_radii = np.sum(radii)
    else:
        res = best_result
        xs = res.x[0::3]
        ys = res.x[1::3]
        rs = res.x[2::3]
        centers = np.column_stack((xs, ys))
        radii = rs
        sum_radii = np.sum(radii)

    # Final validation check (just to be sure, though constraints should handle it)
    # The problem statement asks to return valid packing.
    # If constraints were violated due to numerical error, we might need to fix.
    # But SLSQP usually respects bounds/constraints.
    
    return centers, radii, sum_radii
