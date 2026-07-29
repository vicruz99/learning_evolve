# sol_000039 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 12653929) state=fa5af9ce sum of radii=2.500412 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import random

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]

    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
    
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

def get_max_radius(center, other_centers, other_radii, i_idx):
    """Calculate the maximum possible radius for circle i given its position and other circles."""
    x, y = center
    r_max = min(x, 1 - x, y, 1 - y)
    if r_max < 0:
        return 0.0
    
    for j in range(len(other_centers)):
        if j == i_idx:
            continue
        dx = x - other_centers[j][0]
        dy = y - other_centers[j][1]
        dist = math.sqrt(dx*dx + dy*dy)
        # The radius is limited by the distance to the center of circle j minus r_j
        # i.e., dist >= r_i + r_j  => r_i <= dist - r_j
        limit = dist - other_radii[j]
        if limit < r_max:
            r_max = limit
    return max(0.0, r_max)

def calculate_sum_radii(centers, radii):
    return np.sum(radii)

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    # Function to run optimization from a specific seed
    def optimize(initial_centers, initial_radii, iterations=5000, step_size=0.01, temp_decay=0.999):
        centers = initial_centers.copy()
        radii = initial_radii.copy()
        
        current_sum = calculate_sum_radii(centers, radii)
        
        # Initial radius adjustment to ensure validity
        for i in range(n):
            r = get_max_radius(centers[i], centers, radii, i)
            radii[i] = min(radii[i], r)
            
        current_sum = calculate_sum_radii(centers, radii)
        
        # Random restarts within the optimization loop can help escape local minima,
        # but here we just do a local search with momentum/decay.
        
        for _ in range(iterations):
            # Pick a random circle to move
            i = random.randint(0, n - 1)
            
            # Current radius is constrained by neighbors
            # We want to move center to allow larger radius
            
            # Current radius
            r_curr = radii[i]
            
            # Try moving center
            # Simple random walk
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            
            new_x = centers[i][0] + dx
            new_y = centers[i][1] + dy
            
            # Clip to valid range (strictly inside)
            # We need some margin to fit a radius. 
            # But get_max_radius handles 0 radius.
            # Let's keep centers in [0,1]
            new_x = max(0.0, min(1.0, new_x))
            new_y = max(0.0, min(1.0, new_y))
            
            # Calculate potential max radius at new position
            # We need to temporarily update center to check
            old_pos = centers[i].copy()
            centers[i] = [new_x, new_y]
            
            new_r = get_max_radius(centers[i], centers, radii, i)
            
            # Accept move if it increases the radius of this circle significantly
            # or if it allows global sum to increase?
            # Here we just greedily increase the radius of the moved circle.
            # But changing radius of one might constrain others.
            # A better metric: sum of radii.
            
            # Let's calculate the new radius for the moved circle
            # Note: This doesn't account for the fact that other circles' radii might decrease.
            # But for a local move, this is a proxy.
            
            # To be safe, we only accept if the circle gets bigger or if we accept with probability (simulated annealing)
            # But let's stick to a greedy approach for stability first.
            
            if new_r > r_curr + 1e-6:
                radii[i] = new_r
                # We don't update other radii immediately to keep state consistent, 
                # but in next steps they will be adjusted.
            else:
                # Revert center
                centers[i] = old_pos
                # Try to decrease radius if it's too large (shouldn't happen if we set it correctly)
                # Actually, radii[i] should be consistent with current position.
                # Let's enforce consistency every step.
                pass
            
            # Enforce consistency for circle i
            # Re-calc max radius at current position
            r_limit = get_max_radius(centers[i], centers, radii, i)
            radii[i] = min(radii[i], r_limit)
            
            # Decay step size
            step_size *= 0.9995

        # Final pass to fix all radii to be valid
        for i in range(n):
            radii[i] = get_max_radius(centers[i], centers, radii, i)
            
        return centers, radii

    # Strategy 1: 5x5 Grid + 1
    # 25 circles in 5x5, 1 in gap
    centers1 = []
    radii1 = []
    
    # 5x5 grid
    spacing = 0.2
    start = 0.1
    for r in range(5):
        for c in range(5):
            centers1.append([start + c*spacing, start + r*spacing])
            radii1.append(0.09) # Initial guess
            
    # 26th circle in center gap
    centers1.append([0.5, 0.5])
    radii1.append(0.01)
    
    centers1 = np.array(centers1)
    radii1 = np.array(radii1)
    
    # Optimize
    c_opt, r_opt = optimize(centers1, radii1, iterations=2000, step_size=0.02)
    s = np.sum(r_opt)
    if s > best_sum:
        best_sum = s
        best_centers = c_opt
        best_radii = r_opt

    # Strategy 2: Hexagonal packing
    # Rows of 5, 5, 5, 5, 5, 1? Or 6 rows?
    # Let's try 6 rows of roughly 4-5 circles
    centers2 = []
    radii2 = []
    
    # Hexagonal packing parameters
    # Row height r*sqrt(3)
    # Let's guess r=0.1
    r_guess = 0.1
    h = r_guess * math.sqrt(3)
    
    # Try to pack 26 circles
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # ...
    
    # Actually, let's just generate a dense random pack and optimize
    # Or a specific hex pattern
    
    # 5 rows of 5 circles + 1
    # Hexagonal shift
    y = r_guess
    x_start = r_guess
    row_count = 0
    count = 0
    while count < 26:
        row_len = 5 if row_count % 2 == 0 else 5 # Try 5 for all, might be tight
        # If tight, maybe 4
        if row_count == 5:
            row_len = 1 # Just to fill
            
        # Shift
        shift = r_guess if row_count % 2 != 0 else 0
        
        x = x_start + shift
        for k in range(row_len):
            if count >= 26: break
            centers2.append([x, y])
            radii2.append(0.09)
            x += 2 * r_guess
            count += 1
        
        y += h
        row_count += 1
        if row_count > 6: break # Safety
        
    if len(centers2) < 26:
        # Pad with random
        while len(centers2) < 26:
            centers2.append([random.random(), random.random()])
            radii2.append(0.01)
            
    centers2 = np.array(centers2[:26])
    radii2 = np.array(radii2[:26])
    
    c_opt, r_opt = optimize(centers2, radii2, iterations=2000, step_size=0.02)
    s = np.sum(r_opt)
    if s > best_sum:
        best_sum = s
        best_centers = c_opt
        best_radii = r_opt

    # Strategy 3: Random restart with better initialization
    # Place circles in corners and edges first
    centers3 = []
    radii3 = []
    
    # 4 corners
    corners = [[0.15, 0.15], [0.85, 0.15], [0.15, 0.85], [0.85, 0.85]]
    for p in corners:
        centers3.append(p)
        radii3.append(0.15)
        
    # Fill rest randomly
    for _ in range(22):
        centers3.append([random.uniform(0, 1), random.uniform(0, 1)])
        radii3.append(0.05)
        
    centers3 = np.array(centers3)
    radii3 = np.array(radii3)
    
    c_opt, r_opt = optimize(centers3, radii3, iterations=3000, step_size=0.02)
    s = np.sum(r_opt)
    if s > best_sum:
        best_sum = s
        best_centers = c_opt
        best_radii = r_opt

    # Final validation and safety trim
    # Ensure radii are consistent
    for i in range(n):
        r_lim = get_max_radius(best_centers[i], best_centers, best_radii, i)
        if best_radii[i] > r_lim + 1e-12:
            best_radii[i] = r_lim
            
    # Check for NaN
    if np.isnan(best_centers).any() or np.isnan(best_radii).any():
        # Fallback to a simple valid packing
        centers_fb = np.zeros((26, 2))
        radii_fb = np.zeros(26)
        # 5x5 grid r=0.1
        idx = 0
        for r in range(5):
            for c in range(5):
                centers_fb[idx] = [0.1 + c*0.2, 0.1 + r*0.2]
                radii_fb[idx] = 0.1
                idx += 1
        # 26th circle small
        centers_fb[25] = [0.2, 0.2] # Gap center? No, 0.2,0.2 is center of 4 circles
        # Gap at 0.2, 0.2. Distance to (0.1, 0.1) is sqrt(0.02) ~ 0.141.
        # r <= 0.141 - 0.1 = 0.041
        radii_fb[25] = 0.04
        best_centers = centers_fb
        best_radii = radii_fb
        best_sum = np.sum(radii_fb)

    return best_centers, best_radii, best_sum
