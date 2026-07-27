import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square.
    Uses a hexagonal initial guess followed by local repulsion optimization.
    """
    n = 26
    
    # 1. Initialize positions in a hexagonal grid pattern
    # 5 rows of 5 circles, plus 1 circle at the bottom
    centers = []
    base_r = 0.1014  # Initial guess based on target sum
    
    # Define row configurations: (y_index, x_shift_amount, num_circles)
    # We use vertical spacing approx 1.732 * r (hexagonal)
    # and horizontal spacing 2 * r.
    
    rows_config = [
        (0, 0, 5),   # Row 0: 5 circles, no shift
        (1, 1, 5),   # Row 1: 5 circles, shifted by 1 unit (r)
        (2, 0, 5),   # Row 2: 5 circles, no shift
        (3, 1, 5),   # Row 3: 5 circles, shifted
        (4, 0, 5),   # Row 4: 5 circles, no shift
        (5, 0, 1)    # Row 5: 1 circle (the 26th)
    ]
    
    # Vertical step
    dy = np.sqrt(3) * base_r
    
    for row_idx, shift, count in rows_config:
        y = row_idx * dy + base_r
        
        # If it's the single circle at the bottom, place it centrally if needed, 
        # but let's just place it at the start for the optimizer to move it.
        if count == 1:
            x_start = 0.5 
            # Actually, let's place the last circle in a gap or bottom center
            # The optimizer will move it. Let's place at x=0.5, y=bottom
            centers.append([0.5, y])
            continue
            
        # Standard row generation
        x_start = base_r
        if shift == 1:
            x_start += base_r # Shift by r
            
        for i in range(count):
            x = x_start + i * (2 * base_r)
            centers.append([x, y])
            
    centers = np.array(centers)
    radii = np.ones(n) * base_r

    # 2. Local Optimization (Repulsion Method)
    # Iteratively adjust centers and radii to maximize sum of radii
    
    # To handle the "maximize sum of radii" objective with constraints,
    # we can treat this as finding a configuration where circles are as large as possible.
    # We will perform a simplified optimization:
    # Fix radii slightly above 0.1, try to fit. If overlaps, shrink radii.
    # Actually, better: Start with fixed large radii, optimize positions to minimize overlap.
    # Then adjust radii.
    
    # Let's use a force-based relaxation for positions, keeping radii fixed initially
    # to resolve overlaps, then we can try to increase radii.
    
    current_r = 0.1014
    
    # Optimization parameters
    iterations = 2000
    learning_rate = 0.005
    
    # We will optimize centers for a fixed radius 'current_r' first to ensure validity
    # Then we can try to scale up.
    
    # To be safe and creative, let's perform a gradient ascent on the minimum clearance.
    # But simpler: Just run a repulsion simulation.
    
    # Rescale centers to unit square [0,1]x[0,1] properly
    # The initial y positions might exceed 1.
    # Max y approx 5 * 1.732 * 0.1 + 0.1 approx 0.97. OK.
    # Max x approx 1.0. OK.
    
    # Force-based optimization to separate circles
    for _ in range(iterations):
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # 1. Boundary forces (push inward if too close to edge)
        margin = current_r * 1.1 # Allow a bit of buffer to find space
        for i in range(n):
            x, y = centers[i]
            # Left
            if x < margin:
                forces[i, 0] += (margin - x) * 10
            # Right
            if x > 1 - margin:
                forces[i, 0] -= (x - (1 - margin)) * 10
            # Bottom
            if y < margin:
                forces[i, 1] += (margin - y) * 10
            # Top
            if y > 1 - margin:
                forces[i, 1] -= (y - (1 - margin)) * 10
                
        # 2. Pairwise repulsion forces
        # Only apply if distance < 2*r (overlap or close contact)
        # We want to push them to at least 2*r distance
        min_dist = 2 * current_r
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                
                if dist < min_dist and dist > 1e-6:
                    # Repulsion force magnitude proportional to overlap
                    overlap = min_dist - dist
                    # Force direction
                    direction = diff / dist
                    # Stronger repulsion for smaller distances
                    force_mag = overlap * 10.0 
                    forces[i] += direction * force_mag
                    forces[j] -= direction * force_mag
                elif dist < 1e-6:
                    # If centers coincide, push random
                    forces[i] += np.random.rand(2) - 0.5
                    forces[j] -= np.random.rand(2) - 0.5

        # Update positions
        centers += learning_rate * forces
        
        # Clamp to valid range [r, 1-r]
        centers = np.clip(centers, current_r, 1 - current_r)

    # 3. Post-optimization Radius Adjustment
    # After centers are well-separated, we can try to increase radii uniformly
    # until the first overlap occurs.
    
    # Check max possible uniform radius increase
    min_sep = np.inf
    for i in range(n):
        # Distance to boundaries
        dist_bound = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        if dist_bound < min_sep:
            min_sep = dist_bound
            
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if dist < min_sep:
                min_sep = dist
                
    # The radius can be at most min_sep / 2 (for inter-circle) or min_sep (for boundary)
    # Actually, for boundary, r <= dist_bound. For inter, 2r <= dist => r <= dist/2.
    
    max_r = min_sep / 2 # Start with conservative estimate
    
    # Check boundary constraints strictly
    for i in range(n):
        x, y = centers[i]
        r_candidate = min(x, 1-x, y, 1-y)
        if r_candidate < max_r:
            max_r = r_candidate
            
    # Refine max_r with binary search or simple check
    # Let's verify max_r
    # Actually, min_sep is the tightest constraint.
    # If min_sep comes from boundary, r <= min_sep.
    # If min_sep comes from pair, r <= min_sep / 2.
    
    # We computed min_sep as min(all distances).
    # If the minimum distance is between centers i and j, then 2r <= dist(i,j) -> r <= dist/2.
    # If the minimum distance is from center i to boundary, then r <= dist(i, bound).
    # Our min_sep calculation mixed these.
    # Let's recalculate properly.
    
    limit_r = 1.0
    # Boundary limits
    for i in range(n):
        r_lim = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        if r_lim < limit_r:
            limit_r = r_lim
            
    # Pair limits
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            r_lim = dist / 2
            if r_lim < limit_r:
                limit_r = r_lim
                
    # Set all radii to this limit
    radii = np.ones(n) * limit_r
    
    # 4. Final Validation and Cleanup
    # Ensure no overlaps with a small tolerance
    # If there are any overlaps due to precision, shrink slightly
    while True:
        overlap_found = False
        for i in range(n):
            # Boundary
            x, y = centers[i]
            r = radii[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                radii[i] = min(x, 1-x, y, 1-y)
                overlap_found = True
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                req_dist = radii[i] + radii[j]
                if dist < req_dist - 1e-9:
                    # Shrink equally
                    excess = req_dist - dist
                    shrink = excess / 2 + 1e-9
                    radii[i] -= shrink
                    radii[j] -= shrink
                    overlap_found = True
        
        if not overlap_found:
            break
            
    # Recalculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii