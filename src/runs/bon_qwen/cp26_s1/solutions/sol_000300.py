# sol_000300 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2823a898) state=ebe73592 sum of radii=1.741876 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses numerical optimization starting from a hexagonal packing configuration.
    """
    n_circles = 26
    np.random.seed(42) # For reproducibility

    # Helper function to calculate objective and penalties
    def objective(x):
        # x contains [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
        # Shape: (n_circles * 3,)
        
        # Unpack
        centers = np.zeros((n_circles, 2))
        radii = np.zeros(n_circles)
        
        for i in range(n_circles):
            centers[i] = x[i*3 : i*3 + 2]
            radii[i] = x[i*3 + 2]
            
        # Objective: Maximize sum of radii (so we minimize negative sum)
        score = -np.sum(radii)
        
        # Penalties
        penalty = 0.0
        penalty_weight = 100.0 # Weight for constraints
        
        # 1. Boundary constraints: 0 <= x-r, x+r <= 1, 0 <= y-r, y+r <= 1
        # Equivalent to r <= x <= 1-r and r <= y <= 1-r
        # Or: x-r >= 0, 1-r-x >= 0, y-r >= 0, 1-r-y >= 0
        # And r >= 0
        
        # Check r >= 0
        if np.any(radii < 0):
            penalty += penalty_weight * 1000
            return score + penalty
            
        # Check boundaries
        # x - r >= 0  => violation if x - r < 0
        viol_x1 = np.maximum(0, -centers[:, 0] + radii)
        # 1 - x - r >= 0 => violation if 1 - x - r < 0 => x + r > 1
        viol_x2 = np.maximum(0, centers[:, 0] + radii - 1)
        viol_y1 = np.maximum(0, -centers[:, 1] + radii)
        viol_y2 = np.maximum(0, centers[:, 1] + radii - 1)
        
        boundary_pen = np.sum(viol_x1**2 + viol_x2**2 + viol_y1**2 + viol_y2**2)
        penalty += penalty_weight * boundary_pen
        
        # 2. Overlap constraints: dist(c1, c2) >= r1 + r2
        # dist - (r1 + r2) >= 0
        # Violation if dist < r1 + r2
        
        # Vectorized distance calculation
        # centers shape (N, 2)
        # diff = centers[i] - centers[j]
        # dist = sqrt(sum(diff^2))
        
        # Efficient overlap check
        # Create a matrix of squared distances? Or loop?
        # For N=26, loop is fine.
        
        overlap_pen = 0.0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    diff = min_dist - dist
                    overlap_pen += diff**2
                    
        penalty += penalty_weight * overlap_pen
        
        return score + penalty

    # Initialization: Hexagonal packing
    # Try to fit 26 circles. 
    # Approx radius for 26 equal circles in hex packing?
    # If we fit in 5 rows, height constraint r(2 + 4sqrt(3)) <= 1 => r <= 0.112
    # Width constraint for 5 circles r <= 0.1.
    # Let's start with r=0.09 to be safe and let optimizer expand.
    
    r_init = 0.09
    centers_init = np.zeros((n_circles, 2))
    radii_init = np.full(n_circles, r_init)
    
    # Arrange in rows
    # Rows with 5, 5, 5, 5, 5, 1? Or 5, 4, 5, 4, 5, 2?
    # Let's do a simple grid first, then maybe perturb.
    # 5x5 grid is 25. Add 1.
    
    row_counts = [5, 5, 5, 5, 5, 1] 
    # Total 26.
    # But 6 rows might be tight on height if r is large.
    # Let's try 5 rows with 5, 6, 5, 5, 5? No, 6 circles width.
    # Let's stick to a compact cluster.
    # Hexagonal rows: 5, 5, 5, 5, 5, 1 is okay but 1 is isolated.
    # Better: 6, 5, 5, 5, 5? No width.
    # Let's just use a rectangular grid 5x5 + 1 at center?
    # But they will overlap.
    
    # Let's generate a hexagonal lattice positions
    idx = 0
    r_hex = 0.08 # Start smaller to avoid huge initial penalty
    y = r_hex
    row = 0
    while idx < n_circles:
        # Number of circles in this row
        if row % 2 == 0:
            n_in_row = 5
            x_start = r_hex
        else:
            n_in_row = 4 # Shifted row fits in gaps
            x_start = 2 * r_hex
        
        # Adjust if we run out of circles
        needed = n_in_row
        if idx + needed > n_circles:
            needed = n_circles - idx
            
        for k in range(needed):
            if idx < n_circles:
                centers_init[idx, 0] = x_start + k * (2 * r_hex)
                centers_init[idx, 1] = y
                radii_init[idx] = r_hex
                idx += 1
        
        y += r_hex * math.sqrt(3)
        row += 1
        
    # Flatten parameters
    x0 = np.zeros(n_circles * 3)
    for i in range(n_circles):
        x0[i*3] = centers_init[i, 0]
        x0[i*3+1] = centers_init[i, 1]
        x0[i*3+2] = radii_init[i]
        
    # Optimization
    # Use Powell method as it doesn't require gradients and handles bounds well (though we handle bounds via penalty)
    # Actually, we can add bounds to scipy optimizer to keep r >= 0 and centers in [0,1]
    # But penalty is also there. Bounds help.
    
    bounds = []
    for i in range(n_circles):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)]) # x, y in [0,1], r in [0, 0.5]
        
    # Run optimization
    # maxiter might need to be high
    res = opt.minimize(objective, x0, method='Powell', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9, 'xtol': 1e-9})
    
    best_x = res.x
    
    # Extract results
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        centers[i, 0] = best_x[i*3]
        centers[i, 1] = best_x[i*3+1]
        radii[i] = best_x[i*3+2]
        
    # Post-processing: Clamp radii and centers to ensure validity
    # Sometimes optimizer pushes r slightly negative or outside if penalty not strong enough
    # But with bounds and penalty, it should be fine.
    # Just a sanity check.
    for i in range(n_circles):
        radii[i] = max(0.0, radii[i])
        centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])
        
    # Re-calculate sum
    sum_radii = np.sum(radii)
    
    # If validation fails, we might need to shrink radii slightly?
    # But with penalty method, it should be valid.
    # Let's do a quick check and shrink if overlapping
    # This is a safety net
    # Check overlaps and shrink
    for _ in range(100):
        overlap_found = False
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                required_dist = radii[i] + radii[j]
                if dist < required_dist - 1e-12:
                    # Shrink both
                    shrink = (required_dist - dist) / 2 + 1e-5
                    radii[i] -= shrink
                    radii[j] -= shrink
                    overlap_found = True
        if not overlap_found:
            break
            
    # Check boundaries again after shrinking
    for i in range(n_circles):
        r = radii[i]
        if r < 0: r = 0
        # Clamp center
        cx = np.clip(centers[i,0], r, 1-r)
        cy = np.clip(centers[i,1], r, 1-r)
        centers[i,0] = cx
        centers[i,1] = cy
        radii[i] = r # update if changed? No, r was from loop.
        
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
