import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n_circles = 26
    
    # Define the optimization variables
    # We optimize a vector [x_0, y_0, r_0, x_1, y_1, r_1, ...]
    # However, for better conditioning, it might be better to keep them separate or use a specific ordering.
    # Let's use order: [x_0, ..., x_{n-1}, y_0, ..., y_{n-1}, r_0, ..., r_{n-1}]
    # But standard approach is usually flat list of all params.
    # Let's stick to [x0, y0, r0, x1, y1, r1, ...] for simplicity in indexing.
    
    def objective(params):
        # params shape: (3 * n_circles,)
        # Extract radii (every 3rd element starting from index 2)
        radii = params[2::3]
        # We want to maximize sum of radii, so minimize negative sum
        return -np.sum(radii)

    def get_constraints():
        cons = []
        
        # 1. Boundary constraints: r <= x <= 1-r  =>  x - r >= 0, 1 - x - r >= 0
        # Same for y
        for i in range(n_circles):
            idx_x = i * 3
            idx_y = i * 3 + 1
            idx_r = i * 3 + 2
            
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[idx_x] - p[idx_r]})
            # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[idx_x] - p[idx_r]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[idx_y] - p[idx_r]})
            # 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[idx_y] - p[idx_r]})
            
            # r >= 0 (small epsilon to avoid degenerate circles if possible, but >=0 is fine)
            # cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[idx_r]})

        # 2. Non-overlap constraints: dist_ij >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                idx_xi = i * 3
                idx_yi = i * 3 + 1
                idx_ri = i * 3 + 2
                
                idx_xj = j * 3
                idx_yj = j * 3 + 1
                idx_rj = j * 3 + 2
                
                def non_overlap(p, i=i, j=j):
                    xi, yi, ri = p[idx_xi], p[idx_yi], p[idx_ri]
                    xj, yj, rj = p[idx_xj], p[idx_yj], p[idx_rj]
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    sum_r_sq = (ri + rj)**2
                    return dist_sq - sum_r_sq
                
                cons.append({'type': 'ineq', 'fun': non_overlap})
                
        return cons

    constraints = get_constraints()

    # Helper to generate initial guess (Hexagonal packing)
    def generate_initial_guess():
        # We want to pack n_circles. 
        # Approximate radius for equal circles in square is ~0.1 for 25 circles.
        # For 26, slightly less. Let's start with r=0.05 to be safe and valid.
        r_init = 0.05
        
        centers = []
        # Hexagonal packing logic
        # Rows offset by r horizontally
        # Vertical spacing r * sqrt(3)
        
        # Estimate how many rows/cols fit
        # Width 1, Height 1
        # Let's just generate a grid of points and pick the first n_circles
        
        candidates = []
        y = r_init
        row_idx = 0
        while y <= 1 - r_init + 1e-9:
            x = r_init
            if row_idx % 2 == 1:
                x += r_init # Shift odd rows
            while x <= 1 - r_init + 1e-9:
                candidates.append((x, y))
                x += 2 * r_init
            y += r_init * np.sqrt(3)
            row_idx += 1
            
        # If we don't have enough candidates with this spacing, shrink spacing or add more rows
        # But with r=0.05, we should have plenty.
        # 1/0.05 = 20. Plenty.
        
        # Shuffle to randomize slightly and pick first n_circles?
        # Or just pick the first n_circles.
        # To avoid symmetry issues, maybe shuffle.
        if len(candidates) < n_circles:
            # Fallback to random grid if generation failed (unlikely)
            grid = np.linspace(0.1, 0.9, 6)
            for g_y in grid:
                for g_x in grid:
                    candidates.append((g_x, g_y))
        
        # Select n_circles
        # Let's take the first n_circles, maybe add some noise
        selected_centers = candidates[:n_circles]
        
        # Convert to params: [x0, y0, r0, x1, y1, r1, ...]
        params = []
        for cx, cy in selected_centers:
            params.extend([cx, cy, r_init])
            
        return np.array(params)

    # Run optimization with multiple restarts to find global optimum
    best_val = -np.inf
    best_params = None
    
    # Number of restarts
    n_restarts = 20
    
    for k in range(n_restarts):
        # Generate initial guess
        # Add some random noise to avoid symmetry and local minima
        x0 = generate_initial_guess()
        
        # Add noise to centers
        noise_scale = 0.02
        noise = np.random.normal(0, noise_scale, size=x0.shape)
        # Keep radii constant for now or add small noise?
        # Only noise centers to keep it valid-ish, radii are small so valid.
        x0[::3] += noise[::3] # x coords
        x0[1::3] += noise[1::3] # y coords
        
        # Clamp to bounds [0.01, 0.99] for centers to ensure r=0.05 is valid
        x0[::3] = np.clip(x0[::3], 0.06, 0.94)
        x0[1::3] = np.clip(x0[1::3], 0.06, 0.94)
        
        # Try to optimize
        try:
            res = opt.minimize(objective, x0, method='SLSQP', constraints=constraints, 
                              options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success:
                current_obj = -res.fun # Convert back to sum of radii
                if current_obj > best_val:
                    best_val = current_obj
                    best_params = res.x.copy()
        except Exception as e:
            # If optimization fails, continue
            pass
            
    if best_params is None:
        # Fallback to a simple grid if everything fails
        # 5x5 grid + 1
        centers = []
        grid_pts = np.linspace(0.1, 0.9, 5)
        for y in grid_pts:
            for x in grid_pts:
                centers.append([x, y])
        centers.append([0.5, 0.5]) # Duplicate, but we need 26. 
        # Actually 25 points. Need 26. 
        # Just place last one somewhere valid.
        # But this is a bad fallback.
        # Let's try one more specific init.
        x0 = generate_initial_guess()
        res = opt.minimize(objective, x0, method='SLSQP', constraints=constraints, options={'maxiter': 5000})
        best_params = res.x
        best_val = -res.fun

    # Extract results
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        centers[i, 0] = best_params[i*3]
        centers[i, 1] = best_params[i*3 + 1]
        radii[i] = best_params[i*3 + 2]
        
    sum_radii = np.sum(radii)
    
    # Sort radii for nicer output? Not required but good for debugging.
    # The validation doesn't require sorted.
    
    return centers, radii, sum_radii