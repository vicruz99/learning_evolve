# sol_000006 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4fe936d0) state=75bc8ce3 sum of radii=1.754968 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    Uses a force-directed expansion starting from a hexagonal grid.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # 1. Initialization: Hexagonal Grid
    # A hexagonal pattern allows for denser packing.
    # We estimate a safe initial radius. 
    # For 26 circles, a hex grid with 5-4-5-4-5-3 rows (approx) works.
    # Let's target r ~ 0.085 initially to ensure space.
    r_init = 0.085
    
    # Generate points in a hexagonal lattice
    # Rows are separated by r * sqrt(3)
    # Horizontal spacing is 2 * r
    # Odd rows are shifted by r
    
    row_y = r_init
    row_idx = 0
    count = 0
    
    # We will fill rows until we have n circles
    # We estimate how many rows we might need or just generate a dense grid
    # and take the first n points that fit.
    
    temp_points = []
    y = r_init
    row_num = 0
    
    while count < n:
        # Determine x start based on row parity
        # Even rows (0, 2, ...): start at r
        # Odd rows (1, 3, ...): start at 2r (shifted by r)
        x_start = r_init if row_num % 2 == 0 else 2 * r_init
        
        # Number of circles that fit in this row width 1
        # Width required for k circles: 2*r + (k-1)*2r = 2rk <= 1 -> k <= 1/(2r)
        # But we just place them and check bounds later or place within [0,1]
        
        # Let's just place as many as fit in x direction
        # x coordinates: x_start, x_start + 2r, ...
        # Last circle center x must satisfy x + r <= 1 => x <= 1 - r
        
        x = x_start
        while x + r_init <= 1.0:
            temp_points.append([x, y])
            count += 1
            if count >= n:
                break
            x += 2 * r_init
        
        y += r_init * np.sqrt(3)
        row_num += 1
        # Safety break if y gets too large
        if y + r_init > 1.0 and count < n:
            # If we can't fit in rows, maybe shrink r_init? 
            # But for n=26, r=0.085 should be fine.
            break
            
    if count < n:
        # Fallback to random or grid if logic fails (unlikely)
        # Just create a grid
        step = 1.0 / 6.0
        idx = 0
        for i in range(6):
            for j in range(6):
                if idx < n:
                    centers[idx] = [step * (i + 0.5), step * (j + 0.5)]
                    radii[idx] = 0.05 # Small safe radius
                    idx += 1
    else:
        centers[:n] = np.array(temp_points[:n])
        radii[:] = r_init

    # 2. Iterative Expansion and Relaxation
    # We try to increase radii and resolve collisions
    
    current_r = r_init
    expansion_factor = 1.001 # How much to grow radii each step
    relaxation_iterations = 50 # How many times to push apart per expansion
    total_iterations = 500
    
    # Precompute indices for overlap checks
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    
    for iter in range(total_iterations):
        # Increase radii
        current_r *= expansion_factor
        radii[:] = current_r
        
        # Relaxation phase to resolve overlaps
        for _ in range(relaxation_iterations):
            moved = False
            
            # Check boundary constraints
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                
                # Left wall
                if x - r < 0:
                    centers[i, 0] = r
                    moved = True
                # Right wall
                if x + r > 1:
                    centers[i, 0] = 1 - r
                    moved = True
                # Bottom wall
                if y - r < 0:
                    centers[i, 1] = r
                    moved = True
                # Top wall
                if y + r > 1:
                    centers[i, 1] = 1 - r
                    moved = True
            
            # Check pair overlaps
            for i, j in pairs:
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist_sq = dx*dx + dy*dy
                dist = np.sqrt(dist_sq)
                
                r_sum = radii[i] + radii[j]
                
                if dist < r_sum and dist > 1e-10:
                    # Overlap detected
                    # Push apart
                    overlap = r_sum - dist
                    # Normalize direction
                    nx = dx / dist
                    ny = dy / dist
                    
                    # Displacement split based on radii (smaller moves more)
                    # Simple split: move each by overlap/2
                    move = overlap / 2.0
                    
                    centers[i, 0] -= nx * move
                    centers[i, 1] -= ny * move
                    centers[j, 0] += nx * move
                    centers[j, 1] += ny * move
                    
                    moved = True
            
            if not moved:
                break
        
        # If radii grew significantly, maybe slow down expansion
        if iter > 100:
            expansion_factor = 1.0005
        if iter > 300:
            expansion_factor = 1.0001

    # Final adjustment to ensure strict validity and calculate sum
    # In case of numerical issues, clamp radii slightly
    # But the loop should have handled it.
    # Let's ensure radii are consistent with final centers (just in case)
    # Actually, the loop sets radii to current_r. 
    # If a circle is squeezed, its effective max radius might be lower.
    # But we increased radii uniformly. If it's invalid, the relaxation failed?
    # The relaxation moves centers, but if circles are trapped, radii might be too big.
    # To be safe, we can recalculate max possible radii for final centers?
    # But that requires LP. 
    # Alternatively, just shrink slightly if invalid.
    
    # Let's do a final check and shrink if necessary
    # This is a safeguard
    valid = False
    scale = 1.0
    while not valid and scale > 0.9:
        valid = True
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x < r or x > 1-r or y < r or y > 1-r:
                valid = False
                break
        if valid:
            # Check overlaps
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
                if not valid: break
        
        if not valid:
            scale *= 0.99
            radii *= scale

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
