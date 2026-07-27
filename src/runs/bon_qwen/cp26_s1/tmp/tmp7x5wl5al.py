import numpy as np
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses an iterative expansion algorithm starting from a hexagonal packing seed.
    """
    n = 26
    # We will try to optimize a configuration.
    # A good heuristic is to start with a dense packing and expand.
    
    # Helper to check validity (not strictly needed for internal logic but good for safety)
    def is_valid(centers, radii):
        n = centers.shape[0]
        # Boundary
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        # Overlap
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-9:
                    return False
        return True

    # Strategy: 
    # 1. Initialize with a hexagonal packing of 26 circles with a small radius.
    # 2. Iteratively increase radii and push circles apart to maximize sum.
    
    # Hexagonal packing layout
    # Rows of 5, 5, 5, 5, 5, 1 is 26 circles? No 26.
    # 5, 5, 5, 5, 5 = 25. 1 left.
    # Or 4, 5, 5, 5, 5, 2 = 26.
    # Or 5, 6, 5, 5, 5 = 26.
    
    # Let's try to arrange them in a 6x5 grid but hexagonal.
    # Rows 0, 2, 4 have 5 circles. Rows 1, 3, 5 have 5 circles. (30 circles)
    # We need 26. Let's remove 4 circles from the corners of a 5x6 block?
    # Or just pick the first 26 in a hexagonal list.
    
    centers = []
    # Hexagonal spacing parameters (initial small r)
    # We'll compute coordinates based on a theoretical r, then scale.
    # Let's assume a target r ~ 0.1.
    # Horizontal spacing dx = r * sqrt(3)
    # Vertical spacing dy = r * 1.5 (since row height is r*sqrt(3), but centers are spaced by r*sqrt(3))
    # Actually, in standard hex packing:
    # Row y = i * r * sqrt(3)
    # Row x offsets: even rows 0, odd rows r*sqrt(3)/2? No.
    # Standard:
    # Row i: y = 2r + i * r * sqrt(3) ? No, that's for touching.
    # Let's use relative coordinates.
    # Row height h = sqrt(3)/2 * (2r) = r*sqrt(3).
    # Row width w = sqrt(3) * (2r) = 2r*sqrt(3) for 2 circles?
    # Let's just place them manually.
    
    r_init = 0.05 # Start small
    centers = []
    
    # Try to fit in 6 rows
    # Row 0: 5 circles
    # Row 1: 5 circles
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 1 circle
    # Total 26.
    
    # Coordinates
    # y_step = r_init * math.sqrt(3)
    # x_step = r_init * math.sqrt(3) # for shifted rows?
    # Actually, if circles touch, x_dist = r*sqrt(3).
    
    # Let's define a generic hex grid generator
    def get_hex_centers(r):
        c_list = []
        # 5 rows of 5
        for i in range(5):
            y = 2*r + i * r * math.sqrt(3)
            # Offset for odd/even rows to pack tighter
            if i % 2 == 1:
                x_start = 2*r + r * math.sqrt(3) / 2 # Shifted
            else:
                x_start = 2*r
            
            # Number of circles in this row
            # If shifted, we might fit one less or same?
            # Width of 5 circles: 2r + 4*r*sqrt(3) approx 8.9r
            # Width of shifted 5 circles: 2r + 4*r*sqrt(3) + shift?
            # Actually, center of first circle in shifted row is at r*sqrt(3)/2 + r?
            # Let's just place 5 circles in each row.
            
            # Standard hex:
            # Even row: x = r, r + 2r, r + 4r ... ? No, that's square.
            # Hex: centers at distance 2r.
            # Row 0: (r, r), (r + 2r, r), ...
            # Row 1: (r + r*sqrt(3), r + r*sqrt(3)/2 * 2? No.)
            # Distance between row centers = r*sqrt(3).
            # Horizontal shift = r*sqrt(3)/2? No, shift is r*sqrt(3)/2 if touching?
            # If touching, triangle side is 2r. Height is r*sqrt(3).
            # Horizontal shift is r. Wait.
            # Triangle of 3 touching circles. Centers form equilateral triangle side 2r.
            # Row 0 centers at y=r. Row 1 centers at y = r + r*sqrt(3).
            # x-coordinates: Row 0: r, 3r, 5r... (spacing 2r)
            # Row 1: r + r, 3r + r... (shifted by r).
            # Wait, if shift is r, then dist between (r, r) and (2r, r + r*sqrt(3)) is sqrt(r^2 + 3r^2) = 2r. Correct.
            # So shift is r.
            
            # Let's use this simple hex model:
            # Row i (0-indexed):
            # y = r + i * r * math.sqrt(3)
            # if i is even: x_start = r
            # if i is odd: x_start = 2*r
            # x spacing = 2r
            
            # Check width for 5 circles:
            # Last x = x_start + 4 * 2r = x_start + 8r.
            # Plus radius r at end -> x + r = x_start + 9r.
            # Even row: r + 9r = 10r. (Same as square grid).
            # Odd row: 2r + 9r = 11r. (Wider).
            # This model is not optimal for width.
            
            # Better hex model:
            # Rows packed closely.
            # y_i = r + i * r * math.sqrt(3)
            # x positions should be shifted to center the row or minimize width.
            # For 5 circles, width is 10r (square) or 2r + 4*r*sqrt(3) (hex)?
            # In hex, horizontal distance between touching circles in same row is 2r.
            # So width is always 10r for 5 circles in a row?
            # No, if rows are offset, you can nest them.
            # But circles in the same row must not overlap, so dist >= 2r.
            # So width is at least 10r.
            # Unless... we don't put them in a straight line?
            # But for max radius, straight line is efficient.
            
            # Wait, if width is 10r, and we have 5 rows, height is roughly 5.5r.
            # If r=0.1, width=1, height=0.55. Fits.
            # But we have 26 circles.
            # If we use 5 rows of 5, we use 25 circles. Width 10r.
            # If r > 0.1, width > 1. Fails.
            # So we cannot have a row of 5 circles with r > 0.1 if they are aligned?
            # Wait, if they are not aligned in x?
            # If we arrange them in a hexagonal lattice, the projection on x-axis is compressed?
            # No, the distance between centers is 2r.
            # If you project onto x-axis, the distance is 2r * cos(theta).
            # If rows are tilted?
            # But we are in a square.
            
            # Let's reconsider the width constraint.
            # If we have 5 circles, the minimal bounding box width is 10r (if aligned).
            # Can we do better?
            # If we arrange them in a 'V' shape or something?
            # But generally, for dense packing, rows are straight.
            # However, hexagonal packing allows rows to be closer vertically.
            # But width is still 10r?
            # Wait. In hexagonal packing, the centers of adjacent circles in a row are separated by 2r.
            # So the span is 10r.
            # Is it possible to have span < 10r?
            # Only if circles overlap, which is not allowed.
            # So for 5 circles in a row, we need width >= 10r.
            # Therefore, r <= 0.1 for any configuration with 5 circles in a row aligned horizontally.
            
            # BUT, maybe we don't need 5 circles in a row?
            # Maybe 6 circles in a row with tilt?
            # Or maybe the rows are not horizontal?
            # Rotating the whole packing by 45 degrees?
            # If we rotate a 5x5 grid by 45 degrees, the bounding box increases.
            # So that's bad.
            
            # What if we have rows of 6 circles but tilted?
            # Or maybe the "rows" are not horizontal lines?
            
            # Let's look at the result for n=26 again.
            # If r > 0.1 is possible, how?
            # Maybe the circles are not in straight rows.
            # Or maybe my width assumption is wrong.
            # Width of k circles: min width of convex hull?
            # If we place 5 circles in a circle pattern?
            # No.
            
            # Let's check the constraint again.
            # Dist >= 2r.
            # If we have 5 points in [0,1]x[0,1] with pairwise dist >= 2r.
            # This is a packing problem.
            # For r > 0.1, 2r > 0.2.
            # Can we fit 5 points with dist > 0.2 in a square?
            # Yes, 4 corners + center?
            # Dist corner to center = 0.5 * sqrt(2) * 0.5? No.
            # Square [0,1]. Center (0.5, 0.5). Corner (0,0). Dist sqrt(0.5) ≈ 0.707.
            # Dist between corners (0,0) and (0,1) is 1.
            # Dist between (0,0) and (1,0) is 1.
            # Dist between (0,0) and (0.5, 0.5) is 0.707.
            # So we can fit 5 points with large separation.
            # But we need to fit 26 points.
            
            # The issue with 5x5 grid is that it forces points to be on a grid with spacing 0.2.
            # If we allow random positions, maybe we can pack tighter?
            # Actually, hexagonal packing IS tighter than square grid.
            # But in hexagonal packing, the "rows" are lines.
            # In a row, points are spaced by 2r.
            # So a row of 5 points spans 10r?
            # Wait, in hexagonal packing, the rows are not necessarily parallel to axes?
            # If we rotate the hexagonal lattice, the projection on x-axis changes.
            # If we rotate by 30 degrees, the width might decrease?
            # A row of 5 points with spacing 2r.
            # Length of chain = 8r.
            # If we align this chain along a diagonal?
            # Diagonal length sqrt(2).
            # 8r <= sqrt(2) * width_projection?
            # Actually, the bounding box of a rotated chain.
            # If we place 5 circles in a line tilted by angle theta.
            # Width = 8r * |cos(theta)| + 2r.
            # Height = 8r * |sin(theta)| + 2r.
            # We need Width <= 1 and Height <= 1.
            # 8r |cos(theta)| + 2r <= 1
            # 8r |sin(theta)| + 2r <= 1
            # r (8 |cos| + 2) <= 1
            # r (8 |sin| + 2) <= 1
            # To maximize r, we need to minimize max(8|cos|+2, 8|sin|+2).
            # Minimum of max(cos, sin) is at 45 deg, value sqrt(2)/2 ≈ 0.707.
            # r (8*0.707 + 2) = r (5.656 + 2) = 7.656 r <= 1.
            # r <= 1/7.656 ≈ 0.13.
            # So if we align 5 circles along the diagonal, we can have r ≈ 0.13.
            # But can we fit multiple such rows?
            # Packing 26 circles.
            # Maybe 5 rows of 5 circles, each row tilted?
            # Or a general perturbation.
            
            # So, the strategy should be:
            # 1. Start with a configuration that is not axis-aligned, or allows for larger r.
            # 2. A good seed is a rotated hexagonal lattice or just a dense random packing.
            # 3. Optimize.
            
            pass

    # Let's implement a simple optimizer.
    # Variables: centers (26, 2), radii (26).
    # We can fix radii to be equal for simplicity first?
    # Or allow them to vary. Varying is better.
    
    # Initialization:
    # Random positions? No, too sparse.
    # Grid positions?
    # Let's try a 6x5 grid (30 points) but we only keep 26.
    # Or 5x5 grid (25) + 1.
    # But we suspect we can go higher than 0.1.
    # Let's try to initialize with r=0.09 (safe) and expand.
    
    np.random.seed(42)
    
    # Initialize centers in a grid pattern, slightly perturbed
    # 6 rows, 5 cols? 30 spots. Pick 26.
    # Or 5 rows, 5 cols = 25. Add 1 in middle.
    
    # Let's try 5 rows, 5 cols grid for first 25, plus 1 at (0.5, 0.5)
    # But (0.5, 0.5) is occupied in 5x5 grid?
    # 5x5 grid centers: 0.1, 0.3, 0.5, 0.7, 0.9.
    # (0.5, 0.5) is a center.
    # So maybe shift grid.
    
    # Better seed: Hexagonal packing of 26 circles.
    # We can generate coordinates for hex packing with small r, then scale up.
    
    centers = []
    # 6 rows. 
    # Row 0: 5 circles
    # Row 1: 5 circles
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 1 circle
    
    # Let's use a hex packing generator that handles row shifts.
    r_seed = 0.05
    centers = []
    
    # We will create a list of coordinates relative to a unit square
    # Then scale? No, we need to find absolute coordinates.
    
    # Let's just place them.
    # Row y coords: y0, y1, y2, y3, y4, y5
    # Row x coords: depend on row index (shifted)
    
    # Let's try to fit 5 circles in a row with spacing s_x.
    # And vertical spacing s_y.
    # For hex, s_x = 2r, s_y = r*sqrt(3).
    # Shift = r.
    
    # We will start with r=0.05 and run optimization.
    
    # Generate 26 points
    points = []
    
    # Row 0 (5 circles)
    for i in range(5):
        x = 0.1 + i * 0.2 # Just a guess, will be optimized
        y = 0.1
        points.append([x, y])
    # Row 1 (5 circles)
    for i in range(5):
        x = 0.1 + i * 0.2
        y = 0.1 + 0.1732 # sqrt(3)/10 approx? No.
        # Let's use a tighter packing initially.
        points.append([x, y]) 
    # ... This manual placement is error prone.
    
    # Better: Use a grid of size 6x5 (30 points) and remove 4.
    # Or just random valid packing.
    
    # Let's create a 6x5 grid of points in [0,1]x[0,1].
    # Spacing 1/5.0 = 0.2.
    # Points at (0.1, 0.1), (0.3, 0.1), ...
    # This fits 25.
    # Let's make it 6x5 grid?
    # x in [0.1, 0.9] step 0.2 -> 5 points.
    # y in [0.1, 0.9] step 0.2 -> 5 points.
    # Total 25.
    # We need 26.
    # Add one at (0.5, 0.5)? Overlaps.
    # Add one at (0.1, 0.95)?
    
    # Let's try a denser initialization.
    # 7x4 grid? 28 points.
    # x step 1/6.5?
    # Let's just randomize 26 points in [0,1] and run optimization.
    # But random points might be far apart, limiting r.
    # We want them close.
    
    # Let's use the "circle packing" initialization:
    # Place circles at random positions, if overlap, move.
    # But simpler: Just use a dense grid and perturb.
    
    # Let's try 6 rows of 5 circles, but remove 4.
    # Grid 6 rows, 5 cols.
    # x coords: 0.1, 0.3, 0.5, 0.7, 0.9
    # y coords: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9?
    # If 6 rows, y spacing 0.1?
    # y: 0.15, 0.3, 0.45, 0.6, 0.75, 0.9?
    # If y spacing is small, r must be small.
    
    # Let's rely on the optimizer to find the density.
    # Initialization: 26 circles with r=0.01 at random positions.
    # Then grow r.
    
    centers = np.random.rand(n, 2)
    radii = np.ones(n) * 0.01
    
    # Optimization Loop
    # We want to maximize sum(radii).
    # Constraints: dist >= r_i + r_j, boundary.
    # We can use a simple gradient-like update.
    # Or just a "repulsion" force simulation.
    # Force F_ij = (dist - (r_i + r_j)) / dist * direction.
    # If dist < r_i + r_j, push apart.
    # Also push against walls.
    # And try to increase radii.
    
    # Algorithm:
    # 1. Set a target radius R_target.
    # 2. Check if valid.
    # 3. If not, resolve overlaps.
    # 4. Increase R_target.
    # But radii can be different.
    
    # Better algorithm:
    # Iterative expansion.
    # In each step:
    # 1. Identify overlapping pairs.
    # 2. Move centers to reduce overlap (repulsion).
    # 3. If no overlaps, try to increase radii slightly.
    
    # Since we want to maximize SUM, maybe we should increase all radii by same amount?
    # Or increase smallest radii?
    # Uniform increase is a good proxy for sum increase.
    
    max_iter = 2000
    learning_rate = 0.05
    r_inc = 0.0001
    
    # Initial valid packing?
    # Let's try to fit them in a grid first to ensure validity.
    # 5x5 grid + 1.
    # Grid points:
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    centers = []
    for y in grid_y:
        for x in grid_x:
            centers.append([x, y])
    # 25 circles. Add 1 at (0.5, 0.5) - wait, (0.5, 0.5) is in grid.
    # Add at (0.5, 0.95)?
    centers.append([0.5, 0.95])
    centers = np.array(centers)
    radii = np.ones(26) * 0.08 # Start with r=0.08 (valid for 25 in 5x5 is 0.1, so 0.08 is safe)
    
    # Actually, 0.1 is valid for 25. 0.08 is very safe.
    # Let's start with r=0.05 to be sure.
    radii[:] = 0.05
    
    # Optimization
    for step in range(max_iter):
        # 1. Check overlaps and resolve
        overlap_exists = False
        # Calculate forces
        forces = np.zeros_like(centers)
        
        for i in range(n):
            # Wall repulsion
            x, y = centers[i]
            r = radii[i]
            # Left
            if x - r < 0:
                forces[i, 0] += (x - r) # Push right (if negative, force is negative? No, x-r < 0 -> need to increase x)
                # Actually if x - r < 0, we are outside. We want x >= r.
                # Error = r - x. Force proportional to error.
                forces[i, 0] += (r - x) * 10.0
            # Right
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 10.0
            # Bottom
            if y - r < 0:
                forces[i, 1] += (r - y) * 10.0
            # Top
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 10.0
            
            for j in range(i + 1, n):
                dist_vec = centers[j] - centers[i]
                dist = np.linalg.norm(dist_vec)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    # Force to push apart
                    if dist > 1e-9:
                        direction = dist_vec / dist
                        force_mag = overlap * 5.0 # Stiffness
                        forces[i] -= direction * force_mag
                        forces[j] += direction * force_mag
                    else:
                        # Random push if coincident
                        forces[i] += np.random.randn(2) * 0.01
                        forces[j] -= np.random.randn(2) * 0.01
                    overlap_exists = True
        
        # Update centers
        centers += learning_rate * forces
        
        # Clamp centers to [0, 1] roughly? 
        # The forces should keep them in, but let's clamp to prevent flying out.
        # But if clamped, it creates force.
        # Let's just allow them to move, but the wall forces handle it.
        # However, numerical stability.
        centers = np.clip(centers, 0, 1) 
        # Wait, clipping might cut off force application?
        # Better to apply forces and update, then if outside, push back.
        # But let's stick to forces.
        
        # 2. Try to increase radii
        # If no significant overlap (or overlaps are small), increase radii.
        # Check max overlap
        max_overlap = 0
        for i in range(n):
            # Boundary check
            x, y = centers[i]
            r = radii[i]
            b_overlap = max(0, r - x, x + r - 1, r - y, y + r - 1)
            max_overlap = max(max_overlap, b_overlap)
            
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[j] - centers[i])
                overlap = (radii[i] + radii[j]) - dist
                max_overlap = max(max_overlap, overlap)
        
        if max_overlap < 1e-5:
            # Valid packing, try to grow
            # Increase radii uniformly
            radii += r_inc
            # Or maybe grow more if space allows?
            # Uniform is safe for sum.
        else:
            # If overlap, maybe shrink radii slightly or just let forces resolve
            # Sometimes shrinking helps escape local minima?
            # But usually forces resolve it.
            # We can reduce r_inc if stuck.
            r_inc = max(0.0001, r_inc * 0.99)
            
    # After optimization, ensure validity
    # There might be small overlaps due to discrete steps.
    # We can run a few more steps with zero growth to settle.
    r_inc = 0
    for _ in range(500):
        forces = np.zeros_like(centers)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0: forces[i, 0] += (r - x) * 10.0
            if x + r > 1: forces[i, 0] -= (x + r - 1) * 10.0
            if y - r < 0: forces[i, 1] += (r - y) * 10.0
            if y + r > 1: forces[i, 1] -= (y + r - 1) * 10.0
            
            for j in range(i + 1, n):
                dist_vec = centers[j] - centers[i]
                dist = np.linalg.norm(dist_vec)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if dist > 1e-9:
                        direction = dist_vec / dist
                        forces[i] -= direction * overlap * 10.0
                        forces[j] += direction * overlap * 10.0
        
        centers += learning_rate * forces
        centers = np.clip(centers, 0, 1) # Hard clamp for safety
        
    # Final adjustment: if still invalid, shrink radii slightly
    # Check validity
    valid = True
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            break
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[j] - centers[i])
            if dist < radii[i] + radii[j] - 1e-9:
                valid = False
                break
    
    if not valid:
        # Shrink radii until valid
        # This is a fallback
        shrink_factor = 1.0
        while not valid:
            radii *= 0.99
            shrink_factor *= 0.99
            # Re-check
            valid = True
            for i in range(n):
                x, y = centers[i]
                r = radii[i]
                if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                    valid = False; break
                for j in range(i + 1, n):
                    dist = np.linalg.norm(centers[j] - centers[i])
                    if dist < radii[i] + radii[j] - 1e-9:
                        valid = False; break
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii