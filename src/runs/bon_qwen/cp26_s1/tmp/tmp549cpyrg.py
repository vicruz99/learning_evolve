import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    # Initialize centers in a hexagonal pattern
    centers = np.zeros((n, 2))
    idx = 0
    # Generate hexagonal grid points
    # Approximate spacing to fit 26 circles
    # We'll start with a small radius and grow
    r_init = 0.05
    dx = 2 * r_init
    dy = r_init * math.sqrt(3)
    
    row = 0
    while idx < n:
        for col in range(int(1.2 / dx) + 2): # Heuristic width
            if idx >= n:
                break
            x = col * dx + (row % 2) * (dx / 2)
            y = row * dy
            # Shift to center in square
            centers[idx, 0] = x + 0.05
            centers[idx, 1] = y + 0.05
            idx += 1
        row += 1
    
    # Initialize radii equal
    radii = np.ones(n) * r_init
    
    # Optimization parameters
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = np.sum(radii)
    
    # Simulated Annealing parameters
    temp = 0.1
    min_temp = 1e-5
    cooling_rate = 0.995
    max_iter = 20000
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    for iteration in range(max_iter):
        # Choose a random circle to move or resize
        i = np.random.randint(0, n)
        
        # Decide whether to move center or adjust radius
        if np.random.rand() < 0.3:
            # Try to increase radius slightly
            delta_r = np.random.uniform(0.0001, 0.005)
            new_radii = radii.copy()
            new_radii[i] += delta_r
        else:
            # Try to move center
            delta_pos = np.random.uniform(-0.05, 0.05, 2)
            new_centers = centers.copy()
            new_centers[i] += delta_pos
            new_radii = radii.copy()
        
        # Validate and calculate score
        # We need a function to check validity and score
        # But to save time, we can use a penalty function
        
        # Check boundaries
        valid = True
        penalty = 0
        new_r = new_radii[i]
        
        # Boundary check for circle i
        if new_r < 0:
            valid = False
        else:
            x, y = centers[i] if isinstance(new_centers, type(None)) else new_centers[i]
            if new_centers is not None:
                x, y = new_centers[i]
            
            # Check boundaries
            if x - new_r < 0 or x + new_r > 1 or y - new_r < 0 or y + new_r > 1:
                penalty += (x - new_r if x - new_r < 0 else 0)**2 + \
                           (x + new_r - 1 if x + new_r > 1 else 0)**2 + \
                           (y - new_r if y - new_r < 0 else 0)**2 + \
                           (y + new_r - 1 if y + new_r > 1 else 0)**2
            
            # Check overlaps with all other circles
            for j in range(n):
                if i == j: continue
                xc, yc = new_centers[i] if new_centers is not None else centers[i]
                xj, yj = new_centers[j] if new_centers is not None else centers[j]
                rj = new_radii[j]
                
                dist_sq = (xc - xj)**2 + (yc - yj)**2
                min_dist = new_r + rj
                
                if dist_sq < min_dist**2:
                    penalty += (min_dist - math.sqrt(dist_sq))**2

        # Objective: maximize sum of radii - penalty
        current_sum = np.sum(new_radii)
        score = current_sum - 1000 * penalty # Heavy penalty for violations
        
        old_sum = np.sum(radii)
        old_penalty = 0 # Assume current state is valid (we maintain validity)
        # Actually, we should track penalty of current state too, but let's assume valid start
        # And we only accept moves that are valid or have lower penalty
        
        # Simple Metropolis criterion with energy = -sum + penalty
        # Let's just use validity check for robustness
        
        if penalty < 1e-9: # Valid move
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = new_centers if new_centers is not None else centers
                best_radii = new_radii
                centers = new_centers if new_centers is not None else centers
                radii = new_radii
                # Cooling might be too aggressive if we find better
                temp *= 0.99 
            elif np.random.rand() < math.exp((current_sum - old_sum) / temp):
                centers = new_centers if new_centers is not None else centers
                radii = new_radii
        else:
            # If invalid, maybe accept with low prob if penalty is small?
            # Or just reject. Let's reject to stay valid.
            pass
            
        # Cooling
        temp *= cooling_rate
        if temp < min_temp:
            temp = min_temp

    # Final validation and cleanup
    # Ensure all circles are inside
    for i in range(n):
        r = best_radii[i]
        x, y = best_centers[i]
        # Clamp if slightly out due to float errors
        if x - r < 0: x = r
        if x + r > 1: x = 1 - r
        if y - r < 0: y = r
        if y + r > 1: y = 1 - r
        best_centers[i] = [x, y]
        # If radius makes it impossible, shrink
        if x < r: 
            r = x
        if 1 - x < r:
            r = 1 - x
        if y < r:
            r = y
        if 1 - y < r:
            r = 1 - y
        best_radii[i] = max(0, r)

    # Check overlaps and shrink if necessary (local repair)
    # A simple iterative shrink for overlaps
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = math.sqrt((best_centers[i,0]-best_centers[j,0])**2 + (best_centers[i,1]-best_centers[j,1])**2)
                r_sum = best_radii[i] + best_radii[j]
                if dist < r_sum:
                    # Shrink the larger one or both
                    overlap = r_sum - dist
                    shrink = overlap / 2
                    best_radii[i] -= shrink
                    best_radii[j] -= shrink
                    changed = True
                    # Ensure non-negative
                    best_radii[i] = max(0, best_radii[i])
                    best_radii[j] = max(0, best_radii[j])

    final_sum = np.sum(best_radii)
    return best_centers, best_radii, final_sum