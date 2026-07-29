# sol_000071 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3353d097) state=cdad71fb sum of radii=2.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Initialization: Hexagonal-like packing
    # We try to place circles in rows with alternating shifts to mimic hexagonal packing.
    # This is generally denser than square packing.
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Try to fit 26 circles. 
    # Let's try a pattern: 5-5-5-5-5-1 or similar.
    # Or just a dense random placement refined by a local search?
    # Let's construct a grid that is slightly compressed to fit, then expand.
    
    # A simple hexagonal grid generation
    # Radius estimate for 26 circles in hex packing might be around 0.1?
    # Let's start with r=0.08 to ensure no overlaps initially.
    r_init = 0.08
    
    idx = 0
    y = r_init
    row = 0
    while idx < n:
        # Determine number of circles in this row
        # Alternating rows might have different counts or shifts
        # Width available: 1.0
        # Circles take 2*r width. 
        # With shift r, effective width per circle is 2r? 
        # Actually, centers at x, x+2r, ...
        # If shifted row, centers at x+r, x+3r...
        
        # Let's fit as many as possible
        # Max x is 1.0 - r
        # Min x is r
        # Step 2*r
        
        max_circles_in_row = int((1.0 - 2*r_init) / (2*r_init)) + 1
        # But we can shift.
        # If we shift, we can fit same number usually, unless boundary issues.
        
        # Let's try to fit 5 circles per row mostly.
        count = min(5, n - idx)
        
        # X coordinates
        # Center the row
        total_width = (count - 1) * 2 * r_init
        start_x = (1.0 - total_width) / 2
        
        # If row is even (0, 2, ...), align with left? Or center?
        # Hexagonal packing usually shifts every other row by r.
        if row % 2 == 1:
            start_x += r_init # Shift by radius
            
            # Check bounds
            if start_x < r_init:
                start_x = r_init
            if start_x + (count-1)*2*r_init > 1.0 - r_init:
                # Adjust count or spacing
                count = count - 1
                start_x = (1.0 - (count-1)*2*r_init) / 2 + r_init
        
        for i in range(count):
            if idx >= n: break
            centers[idx, 0] = start_x + i * 2 * r_init
            centers[idx, 1] = y
            radii[idx] = r_init
            idx += 1
            
        y += math.sqrt(3) * r_init # Vertical spacing for hex packing
        row += 1
        
        # Safety break if y goes too high
        if y + r_init > 1.0:
            # Wrap around or just stop? 
            # Better to just fill in a new section?
            # Let's just break and hope we filled enough or reset?
            # Actually if we run out of space, this init is bad.
            # But with r=0.08, height is small.
            # 0.08 * sqrt(3) approx 0.138. 
            # 6 rows fit easily (0.8).
            pass

    # If we didn't fill 26 (unlikely with r=0.08), fill rest randomly?
    # But let's assume the loop filled it. 
    # If not, we can just pad.
    if idx < n:
        # Fallback: place remaining in gaps or just random valid spots
        # But r=0.08 should fit 26 easily.
        # Let's just place remaining in a grid at bottom
        pass

    # Reshape for optimizer
    # Vector: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((1e-6, 0.5)) # r (strictly positive)

    # Objective function
    # Minimize -sum(r) + Penalty
    def objective(vars):
        # vars is 1D array
        cx = vars[0::3]
        cy = vars[1::3]
        cr = vars[2::3]
        
        # Sum of radii (we want to maximize, so minimize negative)
        score = -np.sum(cr)
        
        penalty = 0.0
        penalty_weight = 1000.0 # High weight to enforce constraints
        
        # Boundary constraints
        # r <= x <= 1-r  => x >= r, x <= 1-r
        # r <= y <= 1-r  => y >= r, y <= 1-r
        
        for i in range(n):
            # Left
            if cx[i] < cr[i]:
                penalty += penalty_weight * (cr[i] - cx[i])**2
            # Right
            if cx[i] + cr[i] > 1.0:
                penalty += penalty_weight * (cx[i] + cr[i] - 1.0)**2
            # Bottom
            if cy[i] < cr[i]:
                penalty += penalty_weight * (cr[i] - cy[i])**2
            # Top
            if cy[i] + cr[i] > 1.0:
                penalty += penalty_weight * (cy[i] + cr[i] - 1.0)**2
                
        # Overlap constraints
        # dist >= r_i + r_j
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)
                sum_r = cr[i] + cr[j]
                if dist < sum_r:
                    # Overlap
                    overlap = sum_r - dist
                    penalty += penalty_weight * overlap**2
        
        return score + penalty

    # Run optimization
    # L-BFGS-B is good for bounded problems
    result = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    # Extract results
    final_vars = result.x
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = final_vars[3*i]
        final_centers[i, 1] = final_vars[3*i+1]
        final_radii[i] = final_vars[3*i+2]
        
    # Clean up very small radii if any (though bounds prevent 0)
    # And ensure non-negative
    final_radii = np.maximum(final_radii, 1e-9)
    
    # Validate internally?
    # The optimizer minimizes penalty, so constraints should be soft-satisfied.
    # But with high penalty, they should be satisfied.
    
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# To improve, let's try to run optimization a few times with different initializations?
# But the function is called once.
# I can embed a loop inside run_packing to try multiple starts.

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Try a few strategies
    strategies = 3
    
    for s in range(strategies):
        centers_init = np.zeros((n, 2))
        radii_init = np.zeros(n)
        
        if s == 0:
            # Hexagonal-ish grid
            r_init = 0.09
            idx = 0
            y = r_init
            row = 0
            while idx < n:
                # Try to fit 5 or 6
                # Simple logic
                count = 5
                if row % 2 == 1:
                    count = 4 # Alternating 5-4? 
                    # Actually 5x5 is 25. 26 needs more.
                    # Let's try to fit as many as possible.
                    pass 
                
                # Recalculate count based on space
                # Just place 5 if possible
                # X range [r, 1-r]
                # Width 1-2r
                # Step 2r
                # Max count = floor((1-2r)/2r) + 1 = floor(1/2r - 1) + 1
                max_c = int((1.0 - 2*r_init) / (2*r_init)) + 1
                # Shift for odd rows
                shift = r_init if (row % 2 == 1) else 0
                
                # Adjust count if shift pushes out
                # With shift, x starts at r+shift? No, x starts at r.
                # If shifted, first center at r + shift? 
                # If we shift right by r, first center is r + r = 2r.
                # Last center 2r + (k-1)2r.
                # Check if last + r <= 1.
                
                # Let's just place them centered in the available space for that row type
                # Type 0: aligned with r
                # Type 1: aligned with 2r (shifted by r)
                
                if row % 2 == 0:
                    start_x = r_init
                else:
                    start_x = 2 * r_init # Shifted
                
                # How many fit?
                # Last x = start_x + (k-1)*2r
                # Condition: Last x + r <= 1 => start_x + (k-1)2r + r <= 1
                # start_x + r + (k-1)2r <= 1
                # (k-1)2r <= 1 - (start_x + r)
                # k-1 <= (1 - start_x - r) / 2r
                # k <= (1 - start_x - r)/2r + 1
                
                available_len = 1.0 - (start_x + r_init)
                if available_len < 0:
                    k_max = 0
                else:
                    k_max = int(available_len / (2*r_init)) + 1
                
                k = min(max_c, k_max)
                
                # If k=0, move to next row?
                if k == 0:
                    # Try to reduce r? No, fixed init r.
                    # Just break or move y?
                    # If we can't fit any, we are done?
                    # But we need n circles.
                    # This init logic might fail to place all.
                    # Let's force placement by reducing spacing?
                    # But we want valid init.
                    # Let's just place remaining in a dense grid at bottom if needed.
                    pass
                
                for i in range(k):
                    if idx >= n: break
                    centers_init[idx, 0] = start_x + i * 2 * r_init
                    centers_init[idx, 1] = y
                    radii_init[idx] = r_init
                    idx += 1
                
                if idx >= n: break
                
                y += math.sqrt(3) * r_init
                row += 1
                
            # If not filled, fill remaining randomly in valid spots?
            # Or just a dense grid for remaining
            while idx < n:
                # Place in a grid at bottom?
                # Just place at (0.5, 0.5) with tiny r?
                # Better: place in a separate grid.
                # Grid spacing 0.05
                # Map idx to grid
                gx = 0.05 + (idx % 10) * 0.1
                gy = 0.05 + (idx // 10) * 0.1
                if gx + 0.05 > 1.0: gx = 0.95
                if gy + 0.05 > 1.0: gy = 0.95
                centers_init[idx, 0] = gx
                centers_init[idx, 1] = gy
                radii_init[idx] = 0.02
                idx += 1

        elif s == 1:
            # Random placement with small radii
            np.random.seed(s)
            for i in range(n):
                centers_init[i] = np.random.uniform(0.1, 0.9, 2)
                radii_init[i] = 0.05
        
        elif s == 2:
            # Square grid 5x5 + 1
            # 25 circles at 0.1, 0.3, 0.5, 0.7, 0.9
            # r=0.09 to leave space
            idx = 0
            for i in range(5):
                for j in range(5):
                    if idx < n:
                        centers_init[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                        radii_init[idx] = 0.09
                        idx += 1
            # 26th circle
            if idx < n:
                # Place in center of 4 circles? (0.2, 0.2) etc?
                # Or just (0.5, 0.5) - wait, occupied.
                # Place at (0.5, 0.05)?
                centers_init[idx] = [0.5, 0.05]
                radii_init[idx] = 0.04
                idx += 1
            
            # Fill rest if any
            while idx < n:
                centers_init[idx] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
                radii_init[idx] = 0.02
                idx += 1

        # Prepare x0
        x0 = np.zeros(n * 3)
        for i in range(n):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = radii_init[i]
            
        bounds = [(0, 1), (0, 1), (1e-6, 0.5)] * n

        def objective(vars):
            cx = vars[0::3]
            cy = vars[1::3]
            cr = vars[2::3]
            
            score = -np.sum(cr)
            penalty = 0.0
            pw = 5000.0 # Increased penalty
            
            # Boundary
            # x >= r
            if np.any(cx < cr):
                diff = cr - cx
                diff[diff < 0] = 0
                penalty += pw * np.sum(diff**2)
            # x <= 1-r => x+r <= 1
            if np.any(cx + cr > 1.0):
                diff = (cx + cr) - 1.0
                diff[diff < 0] = 0
                penalty += pw * np.sum(diff**2)
            # y >= r
            if np.any(cy < cr):
                diff = cr - cy
                diff[diff < 0] = 0
                penalty += pw * np.sum(diff**2)
            # y <= 1-r
            if np.any(cy + cr > 1.0):
                diff = (cy + cr) - 1.0
                diff[diff < 0] = 0
                penalty += pw * np.sum(diff**2)
            
            # Overlap
            # Vectorized overlap check is better but loop is fine for n=26
            for i in range(n):
                for j in range(i + 1, n):
                    dx = cx[i] - cx[j]
                    dy = cy[i] - cy[j]
                    dist_sq = dx*dx + dy*dy
                    sum_r = cr[i] + cr[j]
                    # Check overlap: dist < sum_r => dist_sq < sum_r^2
                    # But we need smooth penalty.
                    # Using dist directly might have issues if dist=0?
                    # dist = sqrt(dist_sq)
                    # If dist_sq is very small, dist is small.
                    # If sum_r > dist, penalty.
                    # Let's use squared distance to avoid sqrt?
                    # Constraint: dist^2 >= (r_i + r_j)^2
                    # Violation: (r_i + r_j)^2 - dist^2 > 0
                    val = sum_r**2 - dist_sq
                    if val > 0:
                        penalty += pw * val**2 # Or just val? 
                        # val^2 is C1?
                        # f(u) = max(0, u)^2. Deriv 2u.
                        # Here u = (r_i+r_j)^2 - dist^2.
                        # Smooth enough.
            
            return score + penalty

        res = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 3000, 'ftol': 1e-12})
        
        if res.success or res.fun < -best_sum + 1e-6: # Just checking if we improved?
            # Actually we want max sum, so min -sum.
            # res.fun includes penalty.
            # We should check the actual sum of radii from the result.
            cx = res.x[0::3]
            cy = res.x[1::3]
            cr = res.x[2::3]
            
            # Check validity roughly
            valid = True
            # Check overlaps
            for i in range(n):
                for j in range(i+1, n):
                    d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                    if d < cr[i] + cr[j] - 1e-9:
                        valid = False
                        break
                if not valid: break
            
            # Check bounds
            if valid:
                for i in range(n):
                    if cx[i] < cr[i] - 1e-9 or cx[i] + cr[i] > 1.0 + 1e-9 or \
                       cy[i] < cr[i] - 1e-9 or cy[i] + cr[i] > 1.0 + 1e-9:
                        valid = False
                        break
            
            current_sum = np.sum(cr)
            if valid and current_sum > best_sum:
                best_sum = current_sum
                best_centers = np.array([[cx[i], cy[i]] for i in range(n)])
                best_radii = np.array(cr)
            elif not valid:
                # Even if invalid by strict check, the penalty method might have found a high sum with small violations.
                # But we need a valid solution.
                # Maybe try to repair?
                # For now, keep best valid.
                pass

    # If no valid solution found (unlikely), return a fallback
    if best_centers is None:
        # Fallback: 5x5 grid r=0.1 + 1 small
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                radii[idx] = 0.1
                idx += 1
        centers[idx] = [0.5, 0.5] # Overlap!
        radii[idx] = 0.0 # Make it valid?
        # Actually just return the 25 circles and one valid small one?
        # But n must be 26.
        # Just place 26th at (0.05, 0.05) with r=0.05? Overlap with (0.1, 0.1).
        # Dist sqrt(0.005) approx 0.07. r1+r2 = 0.15. Overlap.
        # Let's just return the optimized one even if slightly invalid?
        # No, must be valid.
        # Let's use the 5x5 grid with r=0.09 and 26th with r=0.04?
        # 5x5 at 0.1, 0.3...
        # 26th at 0.2, 0.2?
        # Dist to (0.1, 0.1) is sqrt(0.02) ~ 0.141.
        # r_25 = 0.09. r_26 = 0.04. Sum = 0.13.
        # 0.141 > 0.13. Valid.
        # But we need to shift 5x5?
        # If 5x5 has r=0.09, centers at 0.1...0.9 is fine.
        # 26th at 0.2, 0.2 with r=0.04.
        # Valid.
        # Sum = 25*0.09 + 0.04 = 2.25 + 0.04 = 2.29.
        # Low, but valid.
        
        # Better fallback:
        # Use the best_centers/radii from last run, but shrink radii to satisfy constraints?
        # Simple shrink: r_i = min(r_i, 0.5 * min_dist)
        # This ensures no overlap.
        # But might be inefficient.
        
        # Let's assume the optimization found a valid or near-valid solution.
        # We will apply a correction step.
        pass

    # Correction step for best_radii/centers if needed
    # This is just a safety net.
    # We will assume the optimizer did a good job with high penalty.
    # But to be safe, let's verify and if overlap, shrink radii.
    
    # Re-validate best solution
    # If best_centers is not set, initialize with something valid.
    if best_centers is None:
        # Initialize valid solution
        centers = np.zeros((26, 2))
        radii = np.zeros(26)
        # Place 25 in grid
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                radii[idx] = 0.09 # Shrink to 0.09
                idx += 1
        # Place 26th
        centers[idx] = [0.5, 0.5] # Center? Overlap.
        # Place at (0.2, 0.2) - gap between (0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.3, 0.3)
        # Dist to (0.1, 0.1) is 0.1414.
        # r_grid = 0.09.
        # Max r_26 = 0.1414 - 0.09 = 0.0514.
        centers[idx] = [0.2, 0.2]
        radii[idx] = 0.05
        best_centers = centers
        best_radii = radii
        best_sum = np.sum(radii)

    # Final check and return
    return best_centers, best_radii, best_sum
