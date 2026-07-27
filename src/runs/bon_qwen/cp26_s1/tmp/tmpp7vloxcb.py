import numpy as np
import math
from scipy.optimize import minimize

def run_packing():
    # Constants
    N = 26
    penalty_weight = 1e4
    boundary_penalty_weight = 1e4
    
    # Helper to compute distance
    def dist(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))

    # Initialize positions: Hexagonal packing attempt
    # We try to fit 26 circles. 
    # A 5x5 grid has 25. We need 1 more.
    # Hexagonal packing allows denser packing.
    # Let's try a layout with 6 rows.
    # Row counts: 5, 5, 5, 5, 4, 2 ? Sum = 26.
    # Or 5, 5, 5, 5, 4, 2 is weird.
    # Maybe 5, 5, 5, 5, 4, 2 is not optimal.
    # Let's try 5 rows of 5 and 1 extra?
    # Better: 6 rows with alternating 5 and 4?
    # 5, 4, 5, 4, 5, 3 -> 26.
    # Let's try to pack them in a rectangle aspect ratio close to 1.
    
    # Layout: 
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 4 circles (shifted)
    # Row 5: 2 circles (shifted) -> Total 26.
    # This seems plausible.
    
    # Initial radius guess
    # For 5 circles in width 1, r = 0.1.
    # For 6 rows hex, height ~ 1 + 5*sqrt(3)/2 * 2r?
    # Height ~ 2r + 5*sqrt(3)r.
    # 2r + 8.66r = 10.66r <= 1 -> r <= 0.0938.
    # Let's start with r = 0.095
    
    r_init = 0.095
    centers = np.zeros((N, 2))
    radii = np.full(N, r_init)
    
    # Construct centers
    idx = 0
    # Row 0: 5 circles
    # y = r
    y = r_init
    for i in range(5):
        centers[idx, 0] = r_init + i * 2 * r_init
        centers[idx, 1] = y
        idx += 1
    
    # Row 1: 5 circles (shifted by r)
    y += math.sqrt(3) * r_init
    x_offset = r_init
    for i in range(5):
        centers[idx, 0] = x_offset + i * 2 * r_init
        centers[idx, 1] = y
        idx += 1
        
    # Row 2: 5 circles
    y += math.sqrt(3) * r_init
    x_offset = 0
    for i in range(5):
        centers[idx, 0] = r_init + i * 2 * r_init # reset to r + i*2r
        centers[idx, 1] = y
        idx += 1
        
    # Row 3: 5 circles (shifted)
    y += math.sqrt(3) * r_init
    x_offset = r_init
    for i in range(5):
        centers[idx, 0] = x_offset + i * 2 * r_init
        centers[idx, 1] = y
        idx += 1
        
    # Row 4: 4 circles
    y += math.sqrt(3) * r_init
    x_offset = 0 # No shift relative to row 2? 
    # To fit 4, we can center them.
    # Span of 4 circles is 6r. Centered at 0.5?
    # Start x = 0.5 - 3r.
    # But let's keep simple grid logic first.
    # If no shift, x = r + i*2r.
    # But we only have 4.
    # Let's just place them.
    for i in range(4):
        centers[idx, 0] = r_init + i * 2 * r_init
        centers[idx, 1] = y
        idx += 1
        
    # Row 5: 2 circles
    y += math.sqrt(3) * r_init
    # Place 2 circles
    for i in range(2):
        centers[idx, 0] = r_init + i * 2 * r_init
        centers[idx, 1] = y
        idx += 1
        
    # Flatten variables: x0, y0, r0, x1, y1, r1, ...
    # Or just x, y arrays and r array.
    # But minimize works with 1D array.
    # Let's pack x, y, r into one vector.
    
    def get_vars():
        return np.concatenate([centers.flatten(), radii])
    
    def set_vars(vars_flat):
        cx = vars_flat[:N*2].reshape((N, 2))
        cr = vars_flat[N*2:]
        return cx, cr

    def objective(vars_flat):
        cx, cr = set_vars(vars_flat)
        
        # Sum of radii
        sum_r = np.sum(cr)
        
        # Penalties
        penalty = 0.0
        
        # Boundary penalties
        for i in range(N):
            x, y = cx[i]
            r = cr[i]
            # Inside [0, 1]
            # x - r >= 0  => r - x <= 0
            if r - x > 0:
                penalty += boundary_penalty_weight * (r - x)**2
            # x + r <= 1 => x + r - 1 <= 0
            if x + r - 1 > 0:
                penalty += boundary_penalty_weight * (x + r - 1)**2
            # y - r >= 0
            if r - y > 0:
                penalty += boundary_penalty_weight * (r - y)**2
            # y + r <= 1
            if y + r - 1 > 0:
                penalty += boundary_penalty_weight * (y + r - 1)**2
        
        # Overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                d = np.sqrt(np.sum((cx[i] - cx[j])**2))
                r_sum = cr[i] + cr[j]
                if r_sum - d > 0:
                    penalty += penalty_weight * (r_sum - d)**2
                    
        return -sum_r + penalty

    # Initial variables
    x0 = get_vars()
    
    # Bounds:
    # x, y in [0, 1]
    # r >= 0
    bounds = []
    for _ in range(N):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 1)) # r
    
    # Run optimization
    # Use L-BFGS-B or SLSQP. L-BFGS-B is faster for large vars.
    # We might need multiple restarts.
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 1000, 'ftol': 1e-9})
    
    # Check result
    cx_opt, cr_opt = set_vars(res.x)
    
    # Post-processing: ensure radii are valid and remove negative/NaN
    cr_opt = np.maximum(cr_opt, 0)
    cx_opt = np.clip(cx_opt, 0, 1)
    
    # Calculate sum
    sum_radii = np.sum(cr_opt)
    
    return cx_opt, cr_opt, sum_radii