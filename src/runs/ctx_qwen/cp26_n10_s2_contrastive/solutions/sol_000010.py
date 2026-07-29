# sol_000010 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b4d6f452) state=a5d42685 sum of radii=2.120008 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def generate_initial_layout(n, width=1.0, height=1.0):
    """
    Generate a hexagonal grid layout for n circles.
    """
    centers = np.zeros((n, 2))
    r_est = 1.0 / (2.0 * np.ceil(np.sqrt(n))) # Initial rough estimate
    
    # Try to fit into a staggered grid
    cols = int(np.ceil(np.sqrt(n * 2 / 1.5))) # Rough guess for columns
    rows = int(np.ceil(n / cols))
    
    # Adjust spacing to fit in unit square
    # Width constraint: cols * diameter <= 1 -> diameter <= 1/cols
    # But in hex, row i has cols, row i+1 might have cols-1 or same
    # Let's assume 'cols' circles in a row
    # Horizontal span: 2*r + (cols-1)*2*r = 2*r*cols
    # Vertical span: 2*r + (rows-1)*r*sqrt(3)
    
    # We want to maximize r. 
    # r <= 1 / (2 * cols)
    # r <= 1 / (2 + (rows-1)*sqrt(3))
    
    # Let's try to find good cols/rows
    best_r = 0
    best_layout = None
    
    for c in range(1, 10):
        for r_idx in range(1, 10):
            if c * r_idx < n: continue
            
            # Width constraint
            w_limit = 1.0 / (2.0 * c)
            # Height constraint (staggered)
            h_limit = 1.0 / (2.0 + (r_idx - 1) * np.sqrt(3))
            
            r_cand = min(w_limit, h_limit)
            
            if r_cand > best_r:
                # Construct this layout
                new_centers = np.zeros((n, 2))
                count = 0
                
                # Alternate row sizes to fit n
                # We have r_idx rows. 
                # Pattern: c, c-1, c, c-1...
                # Or just c circles per row, truncate if needed
                # For hex packing, rows should be offset by r
                
                # Let's stick to 'c' circles per row for simplicity in logic, 
                # but we can offset x.
                
                # Actually, better: define row lengths
                # To maximize density, alternate c and c-1?
                # But for code simplicity, let's just use c columns and offset.
                
                # Re-calculate r based on specific placement
                # Let's just place them and see.
                
                current_r = r_cand
                
                # Generate centers
                # Row y: current_r + row_index * current_r * sqrt(3)
                # Col x: current_r + col_index * 2 * current_r
                # Offset odd rows by current_r
                
                placed = 0
                for row in range(r_idx):
                    y = current_r + row * current_r * np.sqrt(3)
                    offset = current_r if row % 2 == 1 else 0
                    
                    # How many in this row?
                    # If row is offset, it might fit one less?
                    # Max x = 1. Center x = offset + k*2r + r <= 1
                    # k*2r <= 1 - offset - r = 1 - 2r (if offset=r)
                    # k <= (1-2r)/2r = 1/(2r) - 1
                    # Max index k_max. Number of items = k_max + 1.
                    # If r = 1/(2c), 1/(2r) = c. k_max = c-1. Count = c.
                    # So even with offset, we can fit 'c' if width allows?
                    # Wait, if offset=r, first center at 2r. Last at 2r + (c-1)2r + r = (2c+1)r.
                    # If r = 1/(2c), last = (2c+1)/(2c) = 1 + 1/(2c) > 1.
                    # So offset rows with 'c' circles don't fit if r is maxed by width of non-offset row.
                    # But we used min(w_limit, h_limit). 
                    # w_limit was for c circles.
                    # So offset rows might need c-1 circles.
                    
                    # Let's just fill up to n
                    num_in_row = c
                    if row % 2 == 1:
                        # Check if c fits in offset row
                        # Rightmost center x = r + (c-1)*2r + r = 2cr.
                        # Wait, offset row centers: r + r, r + 3r...
                        # First: 2r. Last: 2r + (c-1)2r = 2cr.
                        # Extent: 2cr + r = (2c+1)r.
                        # If r = 1/(2c), extent = 1 + 0.5. Too big.
                        # So offset rows can only fit c-1 circles if r is tight on width.
                        # Let's try c-1.
                        if placed + (c - 1) >= n:
                             num_in_row = n - placed
                        else:
                             num_in_row = c - 1
                    
                    for col in range(num_in_row):
                        if placed >= n: break
                        x = current_r + offset + col * 2 * current_r
                        centers[placed] = [x, y]
                        placed += 1
                    if placed >= n: break
            
            if placed == n:
                best_r = r_cand
                # Copy valid centers (first n)
                if best_layout is None:
                    best_layout = centers[:n].copy()
    
    return best_layout if best_layout is not None else centers

def solve_packing():
    n = 26
    
    # 1. Initialization
    # Use a dense hexagonal-ish layout
    # Manually constructing a good layout for 26
    # 6, 5, 6, 5, 4 rows in hex pattern fits well?
    # Let's use the function
    centers = generate_initial_layout(n)
    
    # Initialize radii small to avoid NaNs or immediate failures
    # Estimate r from layout density
    # Find min distance between any pair and boundaries
    min_dist = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
        
        # Boundaries
        x, y = centers[i]
        min_dist = min(min_dist, x, 1-x, y, 1-y)
        
    radii = np.full(n, min_dist * 0.5) # Start with safe radii
    
    # 2. Expansion Phase (Inflate radii)
    # We iterate: update radii based on current centers
    # r_i = min( boundary_dist, min_j( dist(i,j) - r_j ) )
    # This is a fixed point iteration.
    
    for _ in range(50):
        # Update radii simultaneously based on previous step
        new_radii = np.copy(radii)
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, 1-x, y, 1-y)
            
            for j in range(n):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                max_r = min(max_r, dist - radii[j])
            
            # Ensure non-negative
            if max_r < 0: max_r = 0.0
            
            # Update only if valid (otherwise it's overlapping heavily)
            # But for stability, we can just clamp
            new_radii[i] = max_r
            
        # Check for decrease? No, it should grow or stabilize.
        # Apply
        radii = new_radii

    # 3. Refinement Phase (Move centers to create space)
    # Use a simple force-based relaxation
    # Force = Repulsion from neighbors + Attraction to center?
    # No attraction needed, just keep them in.
    # We want to maximize min radius.
    
    # Gradient ascent on sum of radii?
    # Or just local random moves that increase sum_radii.
    
    best_sum = np.sum(radii)
    
    # Simulated Annealing / Local Search
    temp = 0.1 # Step size
    for iteration in range(2000):
        # Pick a random circle
        idx = np.random.randint(0, n)
        
        # Try to move it
        # Direction: away from closest neighbors?
        # Or just random nudge
        
        dx = (np.random.random() - 0.5) * temp
        dy = (np.random.random() - 0.5) * temp
        
        new_x = centers[idx, 0] + dx
        new_y = centers[idx, 1] + dy
        
        # Check boundary
        if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
            continue
            
        # Calculate potential max radius at new position
        # r_new = min( boundary, min( dist - r_neighbor ) )
        
        r_bound = min(new_x, 1-new_x, new_y, 1-new_y)
        r_min = r_bound
        
        # We need to check against other circles' CURRENT radii
        # Because we are moving one, others stay fixed.
        # If we move idx, its new radius will be constrained by others.
        # We assume others' radii don't change instantly, but for validation,
        # we need r_idx + r_j <= dist.
        # So r_idx <= dist - r_j.
        
        for j in range(n):
            if idx == j: continue
            dist = np.sqrt((new_x - centers[j, 0])**2 + (new_y - centers[j, 1])**2)
            limit = dist - radii[j]
            if limit < r_min:
                r_min = limit
        
        if r_min < 0: # Overlap cannot be resolved by reducing r_idx to 0?
             # Actually if r_min < 0, it means dist < radii[j], so overlap is unavoidable
             # even with r_idx=0.
             # In this case, this move is invalid.
             continue
            
        # Evaluate gain
        # Current contribution: radii[idx]
        # New contribution: r_min
        # But wait, moving idx might allow OTHER circles to grow?
        # For simplicity, we just accept if r_min > radii[idx]
        # This is greedy.
        
        if r_min > radii[idx]:
            centers[idx] = [new_x, new_y]
            radii[idx] = r_min
            # Optional: try to increase neighbors?
            # No, keep it simple.
            
            # Decrease temperature
            if iteration % 100 == 0:
                temp *= 0.9
                
    # 4. Final Expansion Pass
    # After moving, run expansion again to settle radii
    for _ in range(50):
        new_radii = np.copy(radii)
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, 1-x, y, 1-y)
            for j in range(n):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                max_r = min(max_r, dist - radii[j])
            if max_r < 0: max_r = 0.0
            new_radii[i] = max_r
        radii = new_radii

    # Ensure validity for the validator
    # The validator checks dist < r_i + r_j - 1e-12.
    # Our logic ensures dist >= r_i + r_j (since r_i <= dist - r_j).
    # Floating point issues might occur.
    
    # Final validation and correction
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Clamp to square
        if x - r < 0: centers[i, 0] = r
        if x + r > 1: centers[i, 0] = 1 - r
        if y - r < 0: centers[i, 1] = r
        if y + r > 1: centers[i, 1] = 1 - r
        
        # Re-calc radius after clamp
        radii[i] = min(centers[i, 0], 1-centers[i, 0], centers[i, 1], 1-centers[i, 1])
        
        for j in range(n):
            if i == j: continue
            dist = np.linalg.norm(centers[i] - centers[j])
            # If overlap, reduce radius
            required = dist - radii[j]
            if required < radii[i]:
                radii[i] = max(0, required) # Should not happen often if logic correct

    # Run one last expansion to fill up
    for _ in range(100):
         for i in range(n):
            max_r = min(centers[i, 0], 1-centers[i, 0], centers[i, 1], 1-centers[i, 1])
            for j in range(n):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                max_r = min(max_r, dist - radii[j])
            if max_r < 0: max_r = 0
            radii[i] = max_r
            
    return centers, radii, np.sum(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    centers, radii, sum_radii = solve_packing()
    return centers, radii, sum_radii
