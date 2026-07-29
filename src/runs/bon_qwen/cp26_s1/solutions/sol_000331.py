# sol_000331 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 28c61761) state=5bb5923a sum of radii=2.040000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    # 1. Initialization: Hexagonal Grid
    # 26 circles. A 5x5 grid (25) is tight, we add one in a gap.
    # Better to use a hexagonal arrangement for 26.
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Place in a rough hexagonal pattern
    # Row counts for 26 in hex packing: 5, 5, 5, 5, 5, 1 (not great)
    # Or 6, 6, 5, 5, 4? 
    # Let's try a simple dense packing by row
    row_counts = [5, 5, 5, 5, 5, 1] # 26 circles
    # Adjusting for better fit in 1x1
    # Let's just use a 5x6 grid thinned out or similar
    # Actually, just a dense random start inside a smaller box is robust for this optimizer
    
    # Better init: 5x5 grid + 1 in center
    r_init = 0.08
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + i * 0.2, 0.1 + j * 0.2]
            radii[idx] = r_init
            idx += 1
    # 26th circle in a gap
    centers[idx] = [0.2, 0.2] 
    radii[idx] = 0.04
    
    # 2. Optimization: Simulated Annealing with Force-directed packing
    # Objective: Maximize sum of radii subject to constraints
    # We simulate "pressure" from the walls and between circles
    
    current_sum = np.sum(radii)
    best_centers = centers.copy()
    best_radii = radii.copy()
    
    # Parameters
    temp = 0.05 # Initial "temperature" or step size
    decay = 0.995
    min_temp = 1e-6
    max_iter = 5000
    
    # Random seed for reproducibility
    np.random.seed(42)
    
    for step in range(max_iter):
        if temp < min_temp:
            break
            
        # Perturb centers
        delta = temp * np.random.randn(n, 2)
        new_centers = centers + delta
        
        # Project back into square (rough)
        # We'll handle boundary constraints during overlap check
        
        # Try to grow radii slightly
        # Heuristic: Target radius based on local density
        # For now, we just keep radii fixed and move centers to minimize overlap,
        # then in a separate phase we grow radii.
        # Actually, let's do a combined approach.
        
        # 1. Calculate forces
        # Force = repulsion if dist < r_i + r_j
        # Force = push to center if near wall
        forces = np.zeros((n, 2))
        
        # Boundary forces
        for i in range(n):
            x, y = new_centers[i]
            r = radii[i]
            # Push away from walls
            if x < r: forces[i, 0] += (r - x) * 100
            if x > 1 - r: forces[i, 0] -= (x - (1 - r)) * 100
            if y < r: forces[i, 1] += (r - y) * 100
            if y > 1 - r: forces[i, 1] -= (y - (1 - r)) * 100
            
            # Also push towards center to avoid sticking to walls unnecessarily if not optimal?
            # No, walls are good for packing.
            
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                diff = new_centers[i] - new_centers[j]
                dist = np.linalg.norm(diff)
                req_dist = radii[i] + radii[j]
                
                if dist < req_dist and dist > 1e-9:
                    overlap = req_dist - dist
                    # Normalize direction
                    dir_vec = diff / dist
                    # Force proportional to overlap
                    # Stiffer spring for better packing
                    force_mag = overlap * 50.0 
                    forces[i] += dir_vec * force_mag
                    forces[j] -= dir_vec * force_mag
                elif dist == 0:
                    # Prevent division by zero, random push
                    forces[i] += np.random.randn(2) * 0.1
                    forces[j] -= np.random.randn(2) * 0.1

        # Apply forces
        new_centers += forces * (temp * 0.1) # Scale by temp
        
        # Clamp to boundaries [r, 1-r]
        for i in range(n):
            r = radii[i]
            new_centers[i, 0] = np.clip(new_centers[i, 0], r, 1 - r)
            new_centers[i, 1] = np.clip(new_centers[i, 1], r, 1 - r)
            
        centers = new_centers
        
        # Try to increase radii
        # Find the minimum slack
        min_slack = 1.0
        for i in range(n):
            r = radii[i]
            # Distance to boundary
            d_bound = min(centers[i,0] - r, 1 - centers[i,0] - r, 
                          centers[i,1] - r, 1 - centers[i,1] - r)
            min_slack = min(min_slack, d_bound)
            
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                d_pair = dist - (radii[i] + radii[j])
                min_slack = min(min_slack, d_pair)
                
        # If slack is positive, we can grow
        if min_slack > 1e-6:
            grow_factor = 1.0 + (min_slack * 0.5) # Grow by half the available space
            radii *= grow_factor
            
        # Update best if valid (rough check) or if sum increased
        # Check validity
        is_valid = True
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x < r - 1e-7 or x > 1 - r + 1e-7 or y < r - 1e-7 or y > 1 - r + 1e-7:
                is_valid = False
                break
        if is_valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < radii[i] + radii[j] - 1e-7:
                        is_valid = False
                        break
                if not is_valid: break
                
        if is_valid:
            current_sum = np.sum(radii)
            if current_sum > np.sum(best_radii):
                best_centers = centers.copy()
                best_radii = radii.copy()

        temp *= decay
        
    return best_centers, best_radii, np.sum(best_radii)
