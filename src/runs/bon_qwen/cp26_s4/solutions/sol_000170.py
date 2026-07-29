# sol_000170 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 403fd447) state=8d39e513 sum of radii=2.020561 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    """
    n = 26
    
    # --- Phase 1: Initialization ---
    # Generate a hexagonal grid packing as a starting point.
    # We estimate a radius slightly smaller than optimal to ensure initial fit.
    # For n=26, a hexagonal arrangement is denser than a square grid.
    
    # Estimate radius. Area of square = 1.
    # Density of hex packing ~ 0.9069. Boundary effects reduce this.
    # Let's guess r ~ 0.1.
    r_start = 0.095
    
    centers = []
    
    # Try to pack rows. 
    # Row 0 at y = r_start. Centers at x = r_start, r_start + 2r_start, ...
    # Row 1 at y = r_start + sqrt(3)*r_start. Shifted by r_start.
    
    y = r_start
    row_idx = 0
    
    while len(centers) < n:
        # Determine x start for this row. 
        # Even rows (0, 2, ...) start at r_start
        # Odd rows (1, 3, ...) start at 0 (but circle must be inside, so x >= r_start)
        # Actually, to fit in [0,1], x must be >= r and <= 1-r.
        # For odd rows, the optimal shift is usually such that circles sit in valleys.
        # Valley is at x = r_start + r_start = 2*r_start?
        # Let's stick to a simple logic:
        # Even row: x starts at r_start
        # Odd row: x starts at 2*r_start (shifted right by r_start)
        
        if row_idx % 2 == 0:
            x_start = r_start
        else:
            x_start = 2 * r_start # Shifted
        
        x = x_start
        while x <= 1 - r_start:
            if len(centers) < n:
                centers.append([x, y])
            x += 2 * r_start
            
        # Move to next row
        y += np.sqrt(3) * r_start
        row_idx += 1
        
    # Convert to numpy array
    centers = np.array(centers[:n])
    radii = np.array([r_start] * n)
    
    # Flatten variables for optimization: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.hstack([centers.flatten(), radii])
    
    # --- Phase 2: Optimization ---
    
    # Bounds: x, y in [0, 1], r in [0, 0.5] (upper bound for r can be 0.5)
    # Actually r can't be more than 0.5 anyway.
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 1.0)) # r
    bounds = tuple(bounds)

    def objective(x_vec):
        # Unpack
        # x_vec has shape (78,)
        # centers are first 52 elements, radii are last 26
        # Actually interleaved: [x1, y1, r1, x2, y2, r2...]
        
        # Reshape
        params = x_vec.reshape(-1, 3) # (26, 3) -> x, y, r
        
        cx = params[:, 0]
        cy = params[:, 1]
        r = params[:, 2]
        
        # Objective: Maximize sum of radii -> Minimize -sum(r)
        obj_val = -np.sum(r)
        
        # Penalty for overlaps
        penalty = 0.0
        penalty_weight = 1000.0
        
        # Pairwise overlap penalty
        # dist^2 < (r_i + r_j)^2 => overlap
        # We want dist >= r_i + r_j
        # Penalty = max(0, r_i + r_j - dist)^2
        
        # Vectorized computation for speed
        # cx: (n,), compute distance matrix
        # Using broadcasting
        cx_diff = cx[:, None] - cx[None, :]
        cy_diff = cy[:, None] - cy[None, :]
        dist_sq = cx_diff**2 + cy_diff**2
        
        # Lower triangle indices to avoid double counting and self
        i, j = np.tril_indices(n, -1)
        
        r_sum = r[i] + r[j]
        dist = np.sqrt(dist_sq[i, j] + 1e-12) # Avoid div by zero if needed, but sqrt ok
        overlap = np.maximum(0, r_sum - dist)
        penalty += np.sum(overlap**2)
        
        # Boundary penalty
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        # max(0, r - x)^2 + ...
        
        # r - x < 0 is ok.
        # overlap_x_min = max(0, r - x)
        overlap_x_min = np.maximum(0, r - cx)
        overlap_x_max = np.maximum(0, r - (1 - cx))
        overlap_y_min = np.maximum(0, r - cy)
        overlap_y_max = np.maximum(0, r - (1 - cy))
        
        penalty += np.sum(overlap_x_min**2)
        penalty += np.sum(overlap_x_max**2)
        penalty += np.sum(overlap_y_min**2)
        penalty += np.sum(overlap_y_max**2)
        
        return obj_val + penalty_weight * penalty

    # Run optimization
    # L-BFGS-B is suitable for bounds
    res = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-9})
    
    # Extract results
    final_params = res.x.reshape(-1, 3)
    centers_opt = final_params[:, :2]
    radii_opt = final_params[:, 2]
    
    # --- Phase 3: Cleanup ---
    # Ensure strict non-overlap. The optimizer minimizes penalty, 
    # but with finite weight, small overlaps might remain.
    # We can iteratively shrink radii until valid.
    
    # Sort circles? No need.
    # Check validity and shrink if necessary.
    # A simple way: while overlap exists, reduce the radii of overlapping circles slightly.
    # Or just compute the max valid radius for the current centers.
    # But centers might need adjustment too.
    # However, if penalty is low, centers are good.
    
    # Let's do a strict validation loop to reduce radii minimally to satisfy constraints.
    # This is safer than relying purely on the penalty weight.
    
    for _ in range(100):
        valid = True
        # Check overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((centers_opt[i,0]-centers_opt[j,0])**2 + (centers_opt[i,1]-centers_opt[j,1])**2)
                req_dist = radii_opt[i] + radii_opt[j]
                if dist < req_dist - 1e-9:
                    # Overlap. Reduce radii.
                    # Reduce both proportionally? Or just reduce sum?
                    # To maintain max sum, reduce the "slack" amount.
                    # amount = req_dist - dist
                    # We need new_r_i + new_r_j <= dist
                    # current_sum = radii_opt[i] + radii_opt[j]
                    # reduction = current_sum - dist
                    # radii_opt[i] -= reduction / 2
                    # radii_opt[j] -= reduction / 2
                    # But this might cause negative radii or boundary issues.
                    # Just shrink the larger one? Or both.
                    # Simple heuristic: shrink both by half the overlap.
                    overlap = req_dist - dist
                    shrink = overlap / 2 + 1e-6
                    radii_opt[i] -= shrink
                    radii_opt[j] -= shrink
                    valid = False
        
        # Check boundaries
        for i in range(n):
            x, y = centers_opt[i]
            r = radii_opt[i]
            max_r_boundary = min(x, 1-x, y, 1-y)
            if r > max_r_boundary + 1e-9:
                radii_opt[i] = max_r_boundary - 1e-9
                valid = False
        
        if valid:
            break
            
    # Ensure non-negative radii
    radii_opt = np.maximum(radii_opt, 0.0)
    
    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii
