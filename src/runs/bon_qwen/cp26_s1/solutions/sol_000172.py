# sol_000172 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 41bc4cbf) state=c98d1b6d sum of radii=2.622763 correctness=1.0
# stdout(first 200): New best sum: 2.6227627308229073
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import itertools

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
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

def get_objective_grad(vars_flat, n):
    """
    Calculates the objective function (negative sum of radii) and its gradient.
    vars_flat: array of shape [n, 3] flattened. Structure: [x0, y0, r0, x1, y1, r1, ...]
    """
    vars_3d = vars_flat.reshape((n, 3))
    centers = vars_3d[:, :2]
    radii = vars_3d[:, 2]
    
    obj = -np.sum(radii)
    
    # Gradient w.r.t radii is -1
    # Gradient w.r.t centers is 0 for objective
    grad = np.zeros_like(vars_flat)
    grad[2::3] = -1.0 
    
    return obj, grad

def create_constraints(n):
    """
    Returns a list of constraint dictionaries for scipy.optimize.
    """
    constraints = []
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # Equivalent to: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(n):
        # x_i - r_i >= 0
        def constr_x_min(idx=i):
            def fun(vars_flat):
                x = vars_flat[idx * 3]
                r = vars_flat[idx * 3 + 2]
                return x - r
            return fun
        constraints.append({'type': 'ineq', 'fun': constr_x_min()})

        # 1 - x_i - r_i >= 0
        def constr_x_max(idx=i):
            def fun(vars_flat):
                x = vars_flat[idx * 3]
                r = vars_flat[idx * 3 + 2]
                return 1.0 - x - r
            return fun
        constraints.append({'type': 'ineq', 'fun': constr_x_max()})

        # y_i - r_i >= 0
        def constr_y_min(idx=i):
            def fun(vars_flat):
                y = vars_flat[idx * 3 + 1]
                r = vars_flat[idx * 3 + 2]
                return y - r
            return fun
        constraints.append({'type': 'ineq', 'fun': constr_y_min()})

        # 1 - y_i - r_i >= 0
        def constr_y_max(idx=i):
            def fun(vars_flat):
                y = vars_flat[idx * 3 + 1]
                r = vars_flat[idx * 3 + 2]
                return 1.0 - y - r
            return fun
        constraints.append({'type': 'ineq', 'fun': constr_y_max()})

    # Non-overlap constraints: dist(i, j) >= r_i + r_j
    # dist^2 >= (r_i + r_j)^2  => dist^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            def constr_overlap(idx_i=i, idx_j=j):
                def fun(vars_flat):
                    x_i, y_i, r_i = vars_flat[idx_i*3], vars_flat[idx_i*3+1], vars_flat[idx_i*3+2]
                    x_j, y_j, r_j = vars_flat[idx_j*3], vars_flat[idx_j*3+1], vars_flat[idx_j*3+2]
                    
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    r_sum = r_i + r_j
                    return dist_sq - r_sum**2
                return fun
            constraints.append({'type': 'ineq', 'fun': constr_overlap()})

    return constraints

def generate_initial_guess(n, pattern='hex'):
    """
    Generates an initial configuration for centers and radii.
    """
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05) # Small initial radius
    
    if pattern == 'grid':
        # Try to fit in a grid
        # 5x5 grid fits 25. We have 26.
        # Place 25 in grid, 1 in center or random
        count = 0
        for r in range(5):
            for c in range(5):
                if count < n:
                    centers[count, 0] = 0.1 + c * 0.2
                    centers[count, 1] = 0.1 + r * 0.2
                    count += 1
        if count < n:
            # Place remaining near center or random valid spot
            centers[count, 0] = 0.5
            centers[count, 1] = 0.5
            # Reduce radius slightly to ensure valid start
            radii[count] = 0.01
            
    elif pattern == 'hex':
        # Hexagonal packing
        # 6 rows
        rows = [5, 4, 5, 4, 5, 3] # Sum = 26
        r_val = 0.08 # Initial radius guess
        
        y_coord = r_val
        row_idx = 0
        count = 0
        for num_circles in rows:
            x_start = r_val + (0 if row_idx % 2 == 0 else r_val) # Shift odd rows
            for c in range(num_circles):
                if count < n:
                    centers[count, 0] = x_start + c * (2 * r_val)
                    centers[count, 1] = y_coord
                    count += 1
            y_coord += r_val * np.sqrt(3)
            row_idx += 1
            
        # If we didn't fit exactly or pattern differs, fill remaining
        while count < n:
            # Random valid position
            x, y = np.random.uniform(0.1, 0.9, 2)
            # Check if far from others
            valid = True
            for k in range(count):
                d = np.sqrt((x - centers[k,0])**2 + (y - centers[k,1])**2)
                if d < 2 * radii[k] + 0.01:
                    valid = False
                    break
            if valid:
                centers[count] = [x, y]
                count += 1
            else:
                # Just perturb
                centers[count] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
                count += 1

    # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
    vars_flat = np.zeros(n * 3)
    for i in range(n):
        vars_flat[i*3] = centers[i, 0]
        vars_flat[i*3+1] = centers[i, 1]
        vars_flat[i*3+2] = radii[i]
        
    return vars_flat

def run_packing():
    n = 26
    best_sum_radii = -np.inf
    best_vars = None
    best_centers = None
    best_radii = None
    
    # Constraint generation is expensive inside loop, create once? 
    # But constraints are closure-free? No, they capture indices. 
    # We recreate them or pass indices. 
    # Actually, for scipy, creating the list of constraint dicts is fine.
    
    # To speed up, we can define a vectorized constraint function?
    # But SLSQP expects scalar constraint functions or arrays.
    # Let's stick to scalar constraints for simplicity, 26 circles -> ~300 constraints.
    # It might be slow. 
    # Optimization: Use a single function returning an array of constraint violations?
    # SLSQP supports 'fun' returning an array.
    
    def constraints_array(vars_flat):
        vars_3d = vars_flat.reshape((n, 3))
        centers = vars_3d[:, :2]
        radii = vars_3d[:, 2]
        
        vals = []
        
        # Boundary constraints
        # x - r >= 0
        vals.extend(centers[:, 0] - radii)
        # 1 - x - r >= 0
        vals.extend(1.0 - centers[:, 0] - radii)
        # y - r >= 0
        vals.extend(centers[:, 1] - radii)
        # 1 - y - r >= 0
        vals.extend(1.0 - centers[:, 1] - radii)
        
        # Overlap constraints
        # dist^2 - (r_i + r_j)^2 >= 0
        # Vectorized calculation
        # Centers shape (n, 2)
        # Compute pairwise distances squared
        # Use broadcasting
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dist_sq = np.sum(diff**2, axis=2)
        # But this computes all pairs, we only need upper triangle
        
        # Efficient pairwise
        # dist_sq[i, j]
        # r_sum[i, j] = r_i + r_j
        
        # Let's just loop, n=26 is small enough for vectorized inner loop?
        # Or just simple loop.
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (centers[i, 0] - centers[j, 0])**2 + (centers[i, 1] - centers[j, 1])**2
                r_sum = radii[i] + radii[j]
                vals.append(dist_sq - r_sum**2)
                
        return np.array(vals)

    constraint = {'type': 'ineq', 'fun': constraints_array}
    
    # Bounds for variables
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    # r can be up to 0.5 (diameter 1)
    bounds = [(0, 1)] * (n * 3)
    # Tighter bounds for r? r <= 0.5.
    # r bounds are at indices 2, 5, 8...
    for i in range(n):
        bounds[i*3 + 2] = (0, 0.5)
        
    # Run multiple attempts
    patterns = ['grid', 'hex']
    num_restarts = 5
    
    for restart in range(num_restarts):
        for pat in patterns:
            try:
                # Perturb initial guess slightly for diversity
                initial_vars = generate_initial_guess(n, pattern=pat)
                noise = np.random.uniform(-0.01, 0.01, size=initial_vars.shape)
                # Ensure r stays positive
                initial_vars[2::3] += np.abs(noise[2::3]) * 0.1 
                # Clip x, y
                initial_vars[0::3] = np.clip(initial_vars[0::3], 0.05, 0.95)
                initial_vars[1::3] = np.clip(initial_vars[1::3], 0.05, 0.95)
                
                # Optimize
                res = minimize(
                    fun=lambda x: -np.sum(x[2::3]), # Objective: maximize sum radii
                    x0=initial_vars,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraint,
                    options={'maxiter': 2000, 'ftol': 1e-9}
                )
                
                if res.success or (res.fun < -best_sum_radii + 0.001): # Check if improved
                    final_vars = res.x
                    final_centers = final_vars.reshape((n, 3))[:, :2]
                    final_radii = final_vars.reshape((n, 3))[:, 2]
                    
                    # Validate
                    if validate_packing(final_centers, final_radii):
                        current_sum = np.sum(final_radii)
                        if current_sum > best_sum_radii:
                            best_sum_radii = current_sum
                            best_centers = final_centers.copy()
                            best_radii = final_radii.copy()
                            print(f"New best sum: {best_sum_radii}")
            except Exception as e:
                print(f"Optimization failed: {e}")
                continue

    # If no valid packing found (unlikely), return a safe default
    if best_centers is None:
        # Fallback: small circles
        best_centers = np.tile([0.5, 0.5], (n, 1))
        best_radii = np.full(n, 0.001)
        best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, best_sum_radii
