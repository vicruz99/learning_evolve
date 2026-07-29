# sol_000003 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2a4ed9f3) state=4636e037 sum of radii=2.425492 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0 or np.isnan(radii[i]):
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    # 1. Initialization: Hexagonal Packing
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Approximate radius for 26 circles in hexagonal packing
    # Estimate based on area/density, then adjusted by optimization
    r_est = 0.09
    idx = 0
    
    # Generate hexagonal grid points
    # Rows are staggered by r horizontally
    # Vertical spacing is r * sqrt(3)
    row = 0
    while idx < n_circles:
        y = r_est + row * (np.sqrt(3) * r_est)
        if y + r_est > 1.0:
            break
        
        # Determine x offset for this row
        offset = r_est if row % 2 == 1 else 0
        
        x = r_est + offset
        while x + r_est <= 1.0 and idx < n_circles:
            centers[idx] = [x, y]
            radii[idx] = r_est
            idx += 1
            x += 2 * r_est
        row += 1
        
    # If we didn't place all circles (rare with this estimate), place remaining randomly
    if idx < n_circles:
        for i in range(idx, n_circles):
            centers[i] = [np.random.rand(), np.random.rand()]
            radii[i] = 0.01

    # 2. Optimization: Local Search to maximize sum of radii
    # We perform multiple passes of "jiggling" and expanding
    
    for _ in range(10000):
        # Pick a random circle
        i = np.random.randint(0, n_circles)
        
        # Try to increase radius slightly
        current_r = radii[i]
        new_r = current_r * 1.001 # Increase by 0.1%
        
        # Check if new radius is valid with current position
        valid = True
        
        # Check boundaries
        x, y = centers[i]
        if x - new_r < 0 or x + new_r > 1 or y - new_r < 0 or y + new_r > 1:
            valid = False
        
        # Check overlaps
        if valid:
            for j in range(n_circles):
                if i == j: continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < new_r + radii[j] - 1e-12:
                    valid = False
                    break
        
        if valid:
            radii[i] = new_r
        else:
            # If not valid, try to move the circle to a better spot
            # Perturb position
            dx = np.random.uniform(-0.01, 0.01)
            dy = np.random.uniform(-0.01, 0.01)
            new_pos = centers[i] + [dx, dy]
            
            # Keep within bounds
            new_pos[0] = np.clip(new_pos[0], radii[i], 1 - radii[i])
            new_pos[1] = np.clip(new_pos[1], radii[i], 1 - radii[i])
            
            # Check overlaps with new position
            pos_valid = True
            for j in range(n_circles):
                if i == j: continue
                dist = np.sqrt(np.sum((new_pos - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-12:
                    pos_valid = False
                    break
            
            if pos_valid:
                centers[i] = new_pos
                # After moving, try to increase radius again
                radii[i] = current_r * 1.0005

    # Final cleanup: Ensure all radii are consistent with positions
    # We can do a quick pass to tighten or adjust radii
    for i in range(n_circles):
        # Max possible radius for this circle given current positions of others
        max_r = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
        for j in range(n_circles):
            if i == j: continue
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            max_r = min(max_r, dist - radii[j])
        radii[i] = max(max_r, 0.0)

    # Ensure non-negative
    radii = np.maximum(radii, 0.0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
