# sol_000096 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e7a6456) state=7ef5729c sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def calculate_energy(centers, radii, penalty_weight=100.0):
    """
    Calculates the energy of the configuration.
    Energy = -sum(radii) + penalty_weight * (overlap_penalty + boundary_penalty)
    """
    n = len(radii)
    sum_radii = np.sum(radii)
    
    overlap_penalty = 0.0
    # Calculate pairwise distances and check overlaps
    # Using broadcasting for efficiency
    # centers shape (n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (n, n, 2)
    dists = np.sqrt(np.sum(diff**2, axis=2)) # (n, n)
    
    # Radii sums matrix
    r_sums = radii[:, np.newaxis] + radii[np.newaxis, :] # (n, n)
    
    # Overlap amount: positive if overlapping
    overlaps = r_sums - dists
    
    # We only care about i < j to avoid double counting and self-overlap
    # Upper triangle
    upper_tri_indices = np.triu_indices(n, k=1)
    active_overlaps = overlaps[upper_tri_indices]
    
    # Penalty for overlaps (squared)
    overlap_penalty = np.sum(np.maximum(0, active_overlaps)**2)
    
    boundary_penalty = 0.0
    # Check boundary constraints: r <= x <= 1-r  =>  x - r >= 0, 1 - r - x >= 0
    # Same for y
    # Penalties for violations
    # x < r  =>  r - x > 0
    # x > 1-r => x - (1-r) > 0  => x + r - 1 > 0
    
    left_viol = np.maximum(0, radii - centers[:, 0])
    right_viol = np.maximum(0, centers[:, 0] + radii - 1)
    bottom_viol = np.maximum(0, radii - centers[:, 1])
    top_viol = np.maximum(0, centers[:, 1] + radii - 1)
    
    boundary_penalty = np.sum(left_viol**2 + right_viol**2 + bottom_viol**2 + top_viol**2)
    
    return -sum_radii + penalty_weight * (overlap_penalty + boundary_penalty)

def run_packing():
    n_circles = 26
    
    # Function to create initial hexagonal packing
    def get_initial_config(seed=0):
        rng = np.random.default_rng(seed)
        
        # Try to arrange in a hexagonal grid pattern
        # Estimate radius. Area of 26 circles approx 0.8.
        # r approx 0.1.
        # Let's place them in rows.
        # Rows configuration for 26: 6, 5, 6, 5, 4
        # Or just random placement in a grid
        
        # Grid based initialization
        # 5 rows, 5-6 columns
        rows = []
        count = 0
        # Attempt 5 rows
        # Row 0: 6 circles
        # Row 1: 5 circles
        # Row 2: 6 circles
        # Row 3: 5 circles
        # Row 4: 4 circles
        row_counts = [6, 5, 6, 5, 4]
        
        centers = []
        r_est = 0.09 # Initial guess
        
        # Hexagonal spacing
        dx = 2 * r_est
        dy = r_est * math.sqrt(3)
        
        y_offset = r_est + 0.05 # Padding from bottom
        x_start_base = r_est + 0.05
        
        current_y = y_offset
        
        for i, count_in_row in enumerate(row_counts):
            # Offset x for alternating rows
            x_offset = (i % 2) * r_est 
            current_x = x_start_base + x_offset
            
            for _ in range(count_in_row):
                if current_x + r_est <= 1.0:
                    centers.append([current_x, current_y])
                    current_x += dx
            current_y += dy
        
        # If we didn't get 26, or if we went out of bounds, fallback to random/grid
        if len(centers) < n_circles:
            # Fallback to simple grid
            centers = []
            x_step = 1.0 / 6
            y_step = 1.0 / 6
            r_est = 0.08
            cx, cy = x_step/2 + r_est, y_step/2 + r_est
            for r in range(6):
                for c in range(6):
                    centers.append([cx + c * x_step, cy + r * y_step])
                    if len(centers) >= n_circles:
                        break
                if len(centers) >= n_circles:
                    break
        
        centers = np.array(centers[:n_circles])
        
        # Add some noise
        noise = rng.uniform(-0.02, 0.02, size=(n_circles, 2))
        centers = np.clip(centers + noise, 0.01, 0.99)
        
        radii = np.full(n_circles, r_est)
        
        return centers, radii

    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # Try multiple initial configurations
    seeds = [0, 1, 2, 3, 4, 10, 20, 42, 100, 123]
    
    for seed in seeds:
        try:
            centers, radii = get_initial_config(seed)
            
            # Flatten variables for scipy
            # [x0, y0, r0, x1, y1, r1, ...]
            x0 = np.concatenate([centers.flatten(), radii])
            
            # Bounds: x, y in [0, 1], r in [0, 0.5]
            bounds = []
            for i in range(n_circles):
                bounds.append((0, 1)) # x
                bounds.append((0, 1)) # y
                bounds.append((1e-6, 0.5)) # r
            
            # Optimization
            # We maximize sum of radii, so minimize -sum(radii) + penalties
            # Use SLSQP or L-BFGS-B. L-BFGS-B is faster but doesn't handle non-linear constraints well without penalty.
            # We use penalty method in objective.
            
            res = opt.minimize(
                lambda vars: calculate_energy(
                    np.reshape(vars[:-n_circles], (n_circles, 2)), 
                    vars[-n_circles:],
                    penalty_weight=1000.0 # High weight to enforce constraints
                ),
                x0,
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-9}
            )
            
            if res.success:
                centers_opt = np.reshape(res.x[:-n_circles], (n_circles, 2))
                radii_opt = res.x[-n_circles:]
                
                # Validate manually to be sure
                # Check overlaps
                valid = True
                sum_r = np.sum(radii_opt)
                
                # Check boundary
                for i in range(n_circles):
                    x, y = centers_opt[i]
                    r = radii_opt[i]
                    if x < r or x > 1-r or y < r or y > 1-r:
                        valid = False
                        break
                
                if valid:
                    for i in range(n_circles):
                        for j in range(i+1, n_circles):
                            d = np.linalg.norm(centers_opt[i] - centers_opt[j])
                            if d < radii_opt[i] + radii_opt[j] - 1e-9:
                                valid = False
                                break
                        if not valid: break
                
                if valid and sum_r > best_sum_radii:
                    best_sum_radii = sum_r
                    best_centers = centers_opt.copy()
                    best_radii = radii_opt.copy()
                    
        except Exception as e:
            print(f"Optimization failed for seed {seed}: {e}")
            continue

    # If no valid solution found (unlikely), return a safe default
    if best_centers is None:
        # Fallback: small circles in grid
        centers = np.zeros((n_circles, 2))
        radii = np.zeros(n_circles)
        r = 0.05
        x_step = 1.0 / 6
        y_step = 1.0 / 6
        idx = 0
        for r_idx in range(6):
            for c_idx in range(6):
                if idx < n_circles:
                    centers[idx] = [x_step * (c_idx + 0.5), y_step * (r_idx + 0.5)]
                    radii[idx] = r
                    idx += 1
        best_centers = centers
        best_radii = radii
        best_sum_radii = np.sum(radii)

    return best_centers, best_radii, best_sum_radii
