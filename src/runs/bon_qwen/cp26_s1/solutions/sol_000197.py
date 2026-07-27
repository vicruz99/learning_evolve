# sol_000197 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1a220354) state=b0cd2353 sum of radii=2.604595 correctness=1.0
# stdout(first 200): Circles 0 and 1 overlap: dist=0.15942606232136639, r1+r2=0.15942606232352657 Circle 15 at (0.8678212617214867, 0.4319952084159077) with radius 0.1321787382842454 is outside the unit square
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

def get_initial_guess(n_circles):
    """
    Generates an initial configuration of circles in a hexagonal-like pattern.
    """
    centers = []
    # Try to fit circles in rows. 26 circles.
    # A 5x5 grid fits 25. Let's try 6 rows.
    # Pattern: 5, 5, 5, 5, 4, 2 = 26? Or 5, 5, 5, 5, 5, 1?
    # Let's try a balanced hexagonal packing.
    
    # Approximate radius for packing
    # If we have 5 circles in a row, width constraint ~ 0.1
    # Let's start with r=0.09
    r_start = 0.09
    row_counts = [5, 5, 5, 5, 5, 1] # Total 26
    
    # If sum is not 26, adjust
    current_count = 0
    for i, count in enumerate(row_counts):
        if current_count + count > n_circles:
            row_counts[i] = n_circles - current_count
            break
        current_count += count
        
    # Re-construct centers
    row_centers = []
    y = r_start
    for i, count in enumerate(row_counts):
        row = []
        if count == 0: continue
        # Hexagonal offset for odd/even rows
        offset = r_start if i % 2 == 1 else 0.0 # Shift by r for hex packing
        
        # Calculate x positions to center them roughly
        # Width available for centers: 1 - 2*r_start
        # But we just need them inside.
        # For 5 circles, span is 4 * 2r = 8r. 
        # Let's just place them with spacing 2r
        x_start = r_start + offset
        # To center the row, we might need to adjust x_start, but simple placement works for init
        # Let's just place them starting from left margin
        # Actually, better to center them.
        total_width_circles = (count - 1) * 2 * r_start
        margin = (1.0 - total_width_circles - 2 * r_start) / 2.0
        if margin < 0: margin = 0.0
        
        x_start = r_start + margin + offset # offset applied relative to ideal center?
        # Simpler: just place on a grid and let optimizer move them
        # Grid with spacing 0.2
        pass
        
    # Let's use a simpler grid init for robustness
    # 6 rows
    rows = 6
    cols = 5
    # Flatten 26 into grid
    centers_list = []
    r_init = 0.08 # Safe initial radius
    
    # Hexagonal packing coordinates
    # Row i: y = r + i * sqrt(3)*r
    # Col j in row i: x = r + j * 2r + (i%2)*r
    
    y_pos = r_init
    count = 0
    row_idx = 0
    while count < n_circles:
        x_pos = r_init
        col_idx = 0
        # Offset for this row
        if row_idx % 2 == 1:
            x_pos += r_init
            
        while count < n_circles:
            if x_pos + r_init <= 1.0:
                centers_list.append([x_pos, y_pos])
                count += 1
                x_pos += 2 * r_init
                col_idx += 1
            else:
                break
        row_idx += 1
        y_pos += np.sqrt(3) * r_init
        
    # If we didn't get enough, fill rest with random small circles in gaps?
    # The hex loop should get enough if r is small enough.
    # With r=0.08, width 0.16, 6 cols fit? 5*0.16=0.8. 6*0.16=0.96.
    # Height: 5 * 0.138 = 0.69. Fits.
    # So we should get > 26 circles. We take first 26.
    
    if len(centers_list) < n_circles:
        # Fallback to random if hex fails
        np.random.seed(42)
        centers_list = np.random.rand(n_circles, 2) * 0.8 + 0.1
        r_init = 0.05
        
    centers = np.array(centers_list[:n_circles])
    radii = np.full(n_circles, r_init)
    return centers, radii

def objective(params, n_circles):
    """
    Objective function: negative sum of radii
    """
    radii = params[2*n_circles:]
    return -np.sum(radii)

def constraints(params, n_circles):
    """
    Returns a list of constraints for scipy.optimize
    """
    x = params[:n_circles]
    y = params[n_circles:2*n_circles]
    r = params[2*n_circles:]
    
    cons = []
    
    # 1. Boundary constraints
    # x - r >= 0  =>  r - x <= 0 (inequality g(x) >= 0 form: x - r >= 0)
    for i in range(n_circles):
        # x_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*n_circles + i]})
        # 1 - x_i >= r_i  =>  1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i] - p[2*n_circles + i]})
        # y_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[n_circles + i] - p[2*n_circles + i]})
        # 1 - y_i >= r_i
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[n_circles + i] - p[2*n_circles + i]})
        
    # 2. Non-overlap constraints
    # dist(i, j) >= r_i + r_j
    # sqrt((x_i-x_j)^2 + (y_i-y_j)^2) - r_i - r_j >= 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            idx_xi = i
            idx_yi = n_circles + i
            idx_ri = 2*n_circles + i
            
            idx_xj = j
            idx_yj = n_circles + j
            idx_rj = 2*n_circles + j
            
            # To avoid lambda closure issues with complex expressions, use a helper or simple lambda
            # However, lambda with default args is safer in loops
            cons.append({
                'type': 'ineq', 
                'fun': lambda p, i=i, j=j: np.sqrt((p[i]-p[j])**2 + (p[n_circles+i]-p[n_circles+j])**2) - p[2*n_circles+i] - p[2*n_circles+j]
            })
            
    return cons

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    # Get initial guess
    centers_init, radii_init = get_initial_guess(n_circles)
    
    # Flatten parameters: [x1...x26, y1...y26, r1...r26]
    x0 = np.concatenate([centers_init[:, 0], centers_init[:, 1], radii_init])
    
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for _ in range(n_circles):
        bounds.append((0, 1)) # x
    for _ in range(n_circles):
        bounds.append((0, 1)) # y
    for _ in range(n_circles):
        bounds.append((0, 0.5)) # r
        
    cons = constraints(x0, n_circles)
    
    # Run optimization
    # Using SLSQP as it handles constraints well
    best_result = None
    best_sum_radii = -np.inf
    
    # Try a few random restarts to find global optimum
    np.random.seed(123) # For reproducibility
    
    for trial in range(5): # 5 restarts
        if trial > 0:
            # Perturb initial guess
            centers_pert, radii_pert = get_initial_guess(n_circles)
            # Add some noise
            noise_scale = 0.05
            centers_pert[:, 0] += np.random.uniform(-noise_scale, noise_scale, n_circles)
            centers_pert[:, 1] += np.random.uniform(-noise_scale, noise_scale, n_circles)
            centers_pert = np.clip(centers_pert, 0.01, 0.99)
            
            x0_trial = np.concatenate([centers_pert[:, 0], centers_pert[:, 1], radii_pert])
        else:
            x0_trial = x0
            
        try:
            res = minimize(
                objective, 
                x0_trial, 
                args=(n_circles,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < best_sum_radii: # res.fun is negative sum
                 # Extract results
                centers_opt = np.column_stack((res.x[:n_circles], res.x[n_circles:2*n_circles]))
                radii_opt = res.x[2*n_circles:]
                sum_radii = np.sum(radii_opt)
                
                # Validate before updating best
                if validate_packing(centers_opt, radii_opt):
                    if sum_radii > best_sum_radii:
                        best_sum_radii = sum_radii
                        best_result = (centers_opt, radii_opt)
        except Exception as e:
            print(f"Optimization error in trial {trial}: {e}")
            continue
            
    if best_result is None:
        # Fallback to initial guess if optimization failed
        return centers_init, radii_init, np.sum(radii_init)
        
    return best_result[0], best_result[1], best_sum_radii

# Execute and print results
if __name__ == "__main__":
    centers, radii, total_radius = run_packing()
    print(f"Total Sum of Radii: {total_radius:.6f}")
    print(f"Validation: {validate_packing(centers, radii)}")
    print(f"First 5 centers: {centers[:5]}")
    print(f"First 5 radii: {radii[:5]}")
