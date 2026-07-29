# sol_000244 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a213d118) state=5ed7ce03 sum of radii=2.340000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def generate_hexagonal_initial_guess(n_circles):
    """Generates a hexagonal grid of circles inside the unit square."""
    r_est = 0.09
    centers = []
    
    # Hexagonal packing parameters
    row_height = r_est * np.sqrt(3)
    
    row_idx = 0
    while len(centers) < n_circles:
        # Shift odd rows to fit hexagonal pattern
        x_offset = r_est if row_idx % 2 == 1 else 0
        
        # Determine number of circles that fit in this row
        # A row needs width 2*r + (k-1)*2*r + 2*r? No.
        # Centers at x_offset + r, x_offset + 3r, ...
        # x_min = r, x_max = 1 - r
        # First center x = x_offset + r. 
        # Actually, simpler:
        # If even row, centers at r, 3r, 5r...
        # If odd row, centers at 2r, 4r, 6r... (shifted by r)
        
        # Let's place centers
        current_row_centers = []
        # Start x
        start_x = r_est if row_idx % 2 == 0 else 2 * r_est
        # Step
        step_x = 2 * r_est
        
        x = start_x
        while x <= 1 - r_est:
            current_row_centers.append(x)
            x += step_x
            
        y = r_est + row_idx * row_height
        
        # Check if y fits
        if y + r_est > 1:
            # Shift entire grid down to fit if needed, or break
            # For initialization, we just break, optimizer will fix it
            break
            
        for cx in current_row_centers:
            if len(centers) < n_circles:
                centers.append([cx, y])
        
        row_idx += 1
    
    # Pad with random circles if we ran out of space in grid (rare for n=26)
    while len(centers) < n_circles:
        centers.append([0.5, 0.5]) # Fallback
        
    centers = np.array(centers[:n_circles])
    radii = np.full(n_circles, r_est)
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None

    # Initial guess
    centers_init, radii_init = generate_hexagonal_initial_guess(n)
    
    # Combine variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.concatenate([centers_init.flatten(), radii_init])
    
    # Define bounds
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]) # x, y, r
        
    def objective(x):
        radii = x[2::3]
        return -np.sum(radii) # Minimize negative sum

    def constraint_boundary(x):
        # x_i - r_i >= 0  => r_i - x_i <= 0
        # 1 - x_i - r_i >= 0 => x_i + r_i - 1 <= 0
        # Same for y
        c = []
        for i in range(n):
            idx = i * 3
            xi = x[idx]
            yi = x[idx+1]
            ri = x[idx+2]
            # r <= x  => r - x <= 0
            c.append(ri - xi)
            # x + r <= 1 => x + r - 1 <= 0
            c.append(xi + ri - 1.0)
            # r <= y
            c.append(ri - yi)
            # y + r <= 1
            c.append(yi + ri - 1.0)
        return c

    def constraint_overlap(x):
        # dist >= r_i + r_j
        # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
        # (r_i + r_j)^2 - dist^2 <= 0
        c = []
        centers = np.reshape(x[:2*n], (n, 2))
        radii = x[2*n:]
        
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = np.sum((centers[i] - centers[j])**2)
                r_sum = radii[i] + radii[j]
                # Avoid sqrt for performance, but constraint is non-linear
                # r_sum^2 <= dist_sq
                c.append(r_sum**2 - dist_sq)
        return c

    # Use SLSQP
    # Note: Providing constraints as dictionaries
    cons = [
        {'type': 'ineq', 'fun': lambda x: -constraint_boundary(x)}, # ineq expects >= 0, so -c <= 0 => -c >= 0
        {'type': 'ineq', 'fun': lambda x: -constraint_overlap(x)}
    ]
    
    # Run multiple restarts to find global optimum
    for seed in range(5):
        if seed > 0:
            # Perturb initial guess
            noise = np.random.normal(0, 0.02, size=len(x0))
            x_curr = x0 + noise
            # Clip to valid bounds roughly
            x_curr[0::3] = np.clip(x_curr[0::3], 0.01, 0.99)
            x_curr[1::3] = np.clip(x_curr[1::3], 0.01, 0.99)
            x_curr[2::3] = np.clip(x_curr[2::3], 0.01, 0.2)
        else:
            x_curr = x0
            
        try:
            res = minimize(objective, x_curr, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or res.fun < -best_sum_radii:
                # Check validity manually to be safe
                curr_centers = np.reshape(res.x[:2*n], (n, 2))
                curr_radii = res.x[2*n:]
                
                # Quick validation
                valid = True
                # Boundary
                for i in range(n):
                    if curr_radii[i] < -1e-6: valid = False
                    if curr_centers[i,0] < curr_radii[i] - 1e-6 or curr_centers[i,0] > 1 - curr_radii[i] + 1e-6: valid = False
                    if curr_centers[i,1] < curr_radii[i] - 1e-6 or curr_centers[i,1] > 1 - curr_radii[i] + 1e-6: valid = False
                
                if valid:
                    # Overlap
                    for i in range(n):
                        for j in range(i+1, n):
                            dist = np.linalg.norm(curr_centers[i] - curr_centers[j])
                            if dist < curr_radii[i] + curr_radii[j] - 1e-6:
                                valid = False
                                break
                        if not valid: break
                        
                if valid:
                    s = np.sum(curr_radii)
                    if s > best_sum_radii:
                        best_sum_radii = s
                        best_centers = curr_centers
                        best_radii = curr_radii
        except Exception:
            continue

    if best_centers is None:
        # Fallback to initial guess if optimization failed completely
        best_centers = centers_init
        best_radii = radii_init
        best_sum_radii = np.sum(radii_init)

    return best_centers, best_radii, best_sum_radii
