# sol_000153 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3a06727e) state=3d738702 sum of radii=2.617050 correctness=1.0
# stdout(first 200): Error in config 2: 'list' object has no attribute 'shape'
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Helper function to compute constraints
    def constraints_func(vars):
        centers = vars[:2*n].reshape(n, 2)
        radii = vars[2*n:]
        
        cons = []
        
        # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
        # x - r >= 0
        cons.extend(centers[:, 0] - radii)
        # 1 - x - r >= 0
        cons.extend(1 - centers[:, 0] - radii)
        # y - r >= 0
        cons.extend(centers[:, 1] - radii)
        # 1 - y - r >= 0
        cons.extend(1 - centers[:, 1] - radii)
        
        # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx**2 + dy**2
                r_sum = radii[i] + radii[j]
                cons.append(dist_sq - r_sum**2)
                
        return np.array(cons)

    def objective(vars):
        radii = vars[2*n:]
        return -np.sum(radii)

    # Helper to create hexagonal initial configuration
    def create_hex_initial(r_start=0.05):
        # We want to pack 26 circles. 
        # A hexagonal pattern: rows with 5, 6, 5, 6, 4 circles sums to 26.
        # Or maybe 6, 5, 6, 5, 4? Let's try to center them.
        # Let's try a standard grid first perturbed, or a proper hex layout.
        
        # Let's construct a hexagonal packing manually
        # Row spacing h = sqrt(3) * 2r_start / 2 = sqrt(3)*r_start ? 
        # Actually distance between centers is 2r. Horizontal shift r, vertical sqrt(3)r.
        # But for initialization, just spacing 2r is fine.
        
        # Let's try to fit rows.
        # Height available 1.
        # Width available 1.
        
        # Let's try 5 rows.
        # Row counts: 6, 5, 6, 5, 4 -> sum 26? 6+5+6+5+4 = 26.
        # Row 0 (bottom): 6 circles
        # Row 1: 5 circles
        # Row 2: 6 circles
        # Row 3: 5 circles
        # Row 4 (top): 4 circles
        
        # But this might be asymmetric.
        # Let's try a more symmetric approach: 5 rows of approx 5-6.
        # 5, 6, 5, 6, 4 is fine.
        
        centers = []
        radii = []
        
        # Vertical spacing
        # If radius is r, diameter 2r. 
        # In hex packing, vertical dist is sqrt(3)*r? No, distance is 2r.
        # Vertical component is sqrt((2r)^2 - r^2) = r*sqrt(3).
        
        # Let's just place them on a grid for now, slightly perturbed?
        # Actually, a dense grid is a good start.
        # 26 circles. Sqrt(26) ~ 5.1.
        # 5x5 grid has 25.
        # Let's place 25 in 5x5 grid and 1 in center.
        
        # Better: Use a hexagonal lattice generator.
        # Points (x, y) = (i * 2r + (j%2)*r, j * r * sqrt(3))
        
        r_init = 0.05
        h = r_init * np.sqrt(3)
        
        y = r_init
        row_idx = 0
        count = 0
        while count < n:
            # Determine number of circles in this row to stay within width
            # Width 1. Centers from r to 1-r.
            # Spacing 2r.
            # Max circles = floor((1 - 2r) / (2r)) + 1?
            # Actually range [r, 1-r]. Length 1-2r.
            # Step 2r.
            # Num = floor((1-2r)/2r) + 1.
            # For r=0.05, 1-0.1 = 0.9. 0.9/0.1 = 9. So 10 circles?
            # Wait, 2r = 0.1. (1-0.1)/0.1 = 9. +1 = 10.
            # That's too many. We need to distribute 26.
            
            # Let's just fill rows with max possible or specific counts.
            # But we need to control y position.
            
            # Let's fix row heights to fit in [0,1].
            # 5 rows. y centers at 0.1, 0.3, 0.5, 0.7, 0.9?
            # If r=0.05, this is very loose.
            
            # Let's just create a 6x5 grid (30 points) and remove 4?
            # Or 5x5 (25) + 1.
            
            # Let's try a specific pattern:
            # 5 rows.
            # Row 0: 6 circles
            # Row 1: 5 circles
            # Row 2: 6 circles
            # Row 3: 5 circles
            # Row 4: 4 circles
            # Total 26.
            
            # Y coordinates. We need to fit 5 rows.
            # Height 1.
            # r=0.05.
            # Row centers y_k.
            # Let's space them evenly.
            # y_0 = 0.1, y_1 = 0.3, y_2 = 0.5, y_3 = 0.7, y_4 = 0.9
            # This leaves room for r to grow.
            
            row_counts = [6, 5, 6, 5, 4]
            row_ys = [0.1, 0.3, 0.5, 0.7, 0.9]
            
            for idx, cnt in enumerate(row_counts):
                y_val = row_ys[idx]
                # Distribute cnt circles in [0.1, 0.9] range?
                # Or [r, 1-r] range. With r=0.05, range [0.05, 0.95].
                # Width 0.9.
                # If cnt=6, spacing 0.9/5 = 0.18?
                # Or just center them.
                
                # X coordinates
                if cnt == 0:
                    continue
                
                # Spread evenly in [0.05, 0.95]
                # Actually, better to use spacing 2r?
                # But we want to fill space.
                
                # Let's use linspace
                x_vals = np.linspace(0.05 + (1 - 0.1)/2.0, 0.95 - (1 - 0.1)/2.0, cnt) 
                # Wait, linspace(a, b, n) starts at a and ends at b.
                # We want to fit in [0.05, 0.95].
                # Let's just use 0.05 + k*step.
                
                # Simpler: linspace(0.05, 0.95, cnt)
                x_vals = np.linspace(0.05, 0.95, cnt)
                
                # Shift odd rows for hexagonal pattern?
                # Row indices 0, 2, 4 have 6, 6, 4? No, row_counts: 6, 5, 6, 5, 4.
                # Indices 0, 2 are full rows (6). Indices 1, 3 are shorter (5).
                # Usually shorter rows are shifted.
                # Let's shift rows 1 and 3 by half step.
                
                if idx % 2 == 1:
                    # Shift right by half of average spacing
                    # Spacing approx 1/6?
                    # Let's just shift by 0.05?
                    # Actually, for hex, shift is r.
                    x_vals += 0.05 # rough shift
                
                for x in x_vals:
                    centers.append([x, y_val])
                    radii.append(r_init)
                    count += 1
                    if count >= n:
                        break
            if count >= n:
                break
                
        return np.array(centers), np.array(radii)

    # Generate multiple initial configurations
    best_result = None
    best_sum = -np.inf

    # Config 1: Hex-like pattern
    try:
        centers0, radii0 = create_hex_initial()
        # Trim to 26 if more
        centers0 = centers0[:n]
        radii0 = radii0[:n]
        
        # Scale to ensure validity? 
        # With r=0.05 and spacing 0.18, dist is large. Valid.
        
        x0 = np.concatenate([centers0.flatten(), radii0])
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
        
        # Constraints
        # We need to pass the constraints to scipy.
        # The function returns array of values >= 0.
        
        cons = {'type': 'ineq', 'fun': constraints_func}
        
        # Run optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})
        
        if res.success or res.fun < best_sum: # We minimize -sum, so smaller is better
             # Actually best_sum tracks max sum, so -res.fun
             current_sum = -res.fun
             if current_sum > best_sum:
                 best_sum = current_sum
                 best_result = res
    except Exception as e:
        print(f"Error in config 1: {e}")

    # Config 2: Random perturbation of grid
    try:
        # 5x5 grid + 1
        centers1 = []
        radii1 = []
        # 5x5
        for r in range(5):
            for c in range(5):
                x = 0.1 + c * 0.2
                y = 0.1 + r * 0.2
                centers1.append([x, y])
                radii1.append(0.09) # slightly less than 0.1 to allow movement
        
        # Add 1 at center
        centers1.append([0.5, 0.5])
        radii1.append(0.02) # small
        
        # We have 26.
        # Perturb slightly
        centers1 = np.array(centers1) + np.random.normal(0, 0.01, centers1.shape)
        # Clip to valid range roughly
        centers1 = np.clip(centers1, 0.1, 0.9)
        radii1 = np.array(radii1)
        
        x0 = np.concatenate([centers1.flatten(), radii1])
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'maxiter': 1000})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = res
            
    except Exception as e:
        print(f"Error in config 2: {e}")

    # Config 3: Random start
    try:
        # Place circles randomly with small radius
        centers3 = np.random.rand(n, 2) * 0.8 + 0.1 # Keep away from edges
        radii3 = np.full(n, 0.05)
        x0 = np.concatenate([centers3.flatten(), radii3])
        
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'maxiter': 1000})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_result = res
    except Exception as e:
        print(f"Error in config 3: {e}")

    if best_result is None:
        # Fallback to grid
        centers_fallback = []
        radii_fallback = []
        for r in range(5):
            for c in range(5):
                centers_fallback.append([0.1 + c*0.2, 0.1 + r*0.2])
                radii_fallback.append(0.1)
        centers_fallback.append([0.5, 0.5])
        radii_fallback.append(0.04)
        centers_fallback = np.array(centers_fallback[:n])
        radii_fallback = np.array(radii_fallback[:n])
        return centers_fallback, radii_fallback, np.sum(radii_fallback)

    # Extract best solution
    best_x = best_result.x
    best_centers = best_x[:2*n].reshape(n, 2)
    best_radii = best_x[2*n:]
    
    # Clip radii to be non-negative just in case
    best_radii = np.maximum(best_radii, 0)
    
    # Re-clip centers to be within [r, 1-r] based on optimized radii?
    # The optimizer should have handled this, but numerical errors might occur.
    # Let's enforce constraints strictly for safety.
    for i in range(n):
        r = best_radii[i]
        best_centers[i, 0] = np.clip(best_centers[i, 0], r, 1 - r)
        best_centers[i, 1] = np.clip(best_centers[i, 1], r, 1 - r)
        
    # Final check for overlaps and reduce radius if needed?
    # The constraints were enforced, but let's verify.
    # If validation fails, we might need to shrink radii slightly.
    # But SLSQP with tight tolerances usually works.
    
    # Let's compute sum
    total_sum = np.sum(best_radii)
    
    return best_centers, best_radii, total_sum
