import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Solves the circle packing problem for 26 circles in a unit square.
    Maximizes the sum of radii.
    """
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Helper to generate initial hexagonal packing
    def generate_initial_guess(n_circles, r_guess):
        centers = []
        row = 0
        while len(centers) < n_circles:
            y = r_guess + row * r_guess * np.sqrt(3)
            # If row is too high, we might need to adjust, but for init we just place
            # To ensure valid start, we can scale down later if needed.
            # But let's try to keep it within bounds roughly.
            
            # Determine x positions
            # Odd rows shifted by r
            shift = r_guess if row % 2 == 1 else 0.0
            x = r_guess + shift
            
            col = 0
            while x + r_guess <= 1.0 and len(centers) < n_circles:
                centers.append([x, y])
                x += 2 * r_guess
                col += 1
            
            row += 1
        
        return np.array(centers[:n_circles])

    # Strategy: Try a few starting radii and configurations
    # A hexagonal packing of 26 circles fits roughly r ~ 0.09-0.10
    # Let's try starting with a loose packing and letting optimizer tighten/expand.
    
    start_r_candidates = [0.08, 0.09, 0.095]
    
    for r_start in start_r_candidates:
        # Generate centers
        centers_init = generate_initial_guess(n, r_start)
        
        # If we didn't get enough centers (shouldn't happen with small r), pad
        if len(centers_init) < n:
            # Fallback to random
            centers_init = np.random.rand(n, 2) * 0.8 + 0.1
            radii_init = np.ones(n) * 0.05
        else:
            radii_init = np.ones(n) * r_start

        # Flatten for optimizer
        # Vector: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = radii_init[i]

        # Define objective: maximize sum(r) => minimize -sum(r)
        def objective(vars_vec):
            radii = vars_vec[2::3]
            return -np.sum(radii)

        # Define constraints
        def constraints(vars_vec):
            constraints_list = []
            x = vars_vec[0::3]
            y = vars_vec[1::3]
            r = vars_vec[2::3]
            
            # Boundary constraints: x-r >= 0 => r-x <= 0
            # x+r <= 1 => x+r-1 <= 0
            # y-r >= 0 => r-y <= 0
            # y+r <= 1 => y+r-1 <= 0
            
            # We can return a single array for inequality constraints g(x) <= 0
            # Or a list of constraint dicts. 
            # SLSQP accepts a list of dicts or a single array if simple.
            # Let's build a list of arrays for clarity or one big array.
            # One big array is faster.
            
            # Boundary
            c_boundary = np.zeros(4 * n)
            c_boundary[0::4] = r - x          # r - x <= 0  (x >= r)
            c_boundary[1::4] = x + r - 1      # x + r - 1 <= 0 (x <= 1-r)
            c_boundary[2::4] = r - y          # r - y <= 0  (y >= r)
            c_boundary[3::4] = y + r - 1      # y + r - 1 <= 0 (y <= 1-r)
            
            # Overlap constraints: dist^2 >= (r_i + r_j)^2
            # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
            # => -( (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 ) <= 0
            # => (r_i + r_j)^2 - ( (x_i - x_j)^2 + (y_i - y_j)^2 ) <= 0
            
            # Number of pairs
            n_pairs = n * (n - 1) // 2
            c_overlap = np.zeros(n_pairs)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                    r_sum_sq = (r[i] + r[j])**2
                    c_overlap[idx] = r_sum_sq - dist_sq
                    idx += 1
            
            # Combine
            all_constraints = np.concatenate([c_boundary, c_overlap])
            return all_constraints

        # SLSQP constraints specification
        cons = {'type': 'ineq', 'fun': lambda vars: -constraints(vars)} # Wait, SLSQP expects g(x) >= 0 for 'ineq'?
        # Documentation: 'ineq' means fun(x) >= 0.
        # My formulation: (r_i + r_j)^2 - dist^2 <= 0.
        # So I need dist^2 - (r_i + r_j)^2 >= 0.
        # Let's redefine constraint function to return value >= 0.
        
        def constraints_geq(vars_vec):
            x = vars_vec[0::3]
            y = vars_vec[1::3]
            r = vars_vec[2::3]
            
            # Boundary: x >= r => x - r >= 0
            c1 = x - r
            c2 = (1 - r) - x # 1 - r - x >= 0
            c3 = y - r
            c4 = (1 - r) - y
            
            # Overlap: dist^2 - (r_i + r_j)^2 >= 0
            c_overlap = []
            for i in range(n):
                for j in range(i + 1, n):
                    dist_sq = (x[i] - x[j])**2 + (y[i] - y[j])**2
                    r_sum_sq = (r[i] + r[j])**2
                    c_overlap.append(dist_sq - r_sum_sq)
            
            return np.concatenate([c1, c2, c3, c4, np.array(c_overlap)])

        cons_dict = {'type': 'ineq', 'fun': constraints_geq}

        # Bounds
        # x, y in [0, 1], r in [0, 1]
        bounds = [(0, 1)] * (2 * n) + [(0, 1)] * n 
        # Actually r can be up to 0.5, but 1 is safe.
        
        # Run optimizer
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons_dict, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or res.fun < -1e-5: # Just check if we got something
                # Check validity explicitly
                centers_opt = res.x[0::3].reshape(n, 2)
                radii_opt = res.x[2::3]
                
                # Clean up small negative radii due to numerical errors
                radii_opt = np.maximum(radii_opt, 1e-9)
                
                # Clip centers to valid range based on radii
                # x must be in [r, 1-r]
                centers_opt[:, 0] = np.clip(centers_opt[:, 0], radii_opt, 1 - radii_opt)
                centers_opt[:, 1] = np.clip(centers_opt[:, 1], radii_opt, 1 - radii_opt)
                
                # Validate
                valid = True
                for i in range(n):
                    if centers_opt[i, 0] - radii_opt[i] < -1e-9 or centers_opt[i, 0] + radii_opt[i] > 1 + 1e-9:
                        valid = False
                        break
                    if centers_opt[i, 1] - radii_opt[i] < -1e-9 or centers_opt[i, 1] + radii_opt[i] > 1 + 1e-9:
                        valid = False
                        break
                
                if valid:
                    current_sum = np.sum(radii_opt)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = centers_opt.copy()
                        best_radii = radii_opt.copy()
        except Exception as e:
            # Fallback or continue
            pass

    # If optimizer failed or sum is low, try a simple equal radius scaling on the best config
    if best_centers is not None:
        # Try to uniformly scale up radii
        # Find the tightest constraint
        # But this is complex. The optimizer should have done it.
        
        # However, SLSQP might get stuck. 
        # Let's try a post-processing step: 
        # If valid, try to increase radii slightly until constraints hit.
        # But we can just trust the optimizer with maxiter 1000.
        
        # Let's ensure we return the best found.
        # If best_sum is still low (init value), we might have a problem.
        # But with r_start=0.08, sum should be around 2.08.
        
        # If best_centers is None, provide a safe fallback
        if best_sum < 1.0:
            # Fallback: 26 circles radius 0.05 in grid
            fallback_radii = np.ones(n) * 0.05
            fallback_centers = np.zeros((n, 2))
            idx = 0
            for r_idx in range(5):
                for c_idx in range(6):
                    if idx < n:
                        fallback_centers[idx] = [0.1 + c_idx * 0.16, 0.1 + r_idx * 0.18]
                        idx += 1
            best_centers = fallback_centers
            best_radii = fallback_radii
            best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum