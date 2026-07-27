import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Uses a growing circles heuristic with overlap resolution.
    """
    n = 26
    # Initialize centers in a grid pattern
    # We want to place 26 circles. 5x6 grid has 30 spots. 
    # Let's place them to cover the square uniformly.
    # 5 columns, 6 rows? 
    # Or just fill a grid.
    
    # Let's try a grid of 5 columns and ceil(26/5) = 6 rows
    # But we only place 26.
    # Grid step: x in [0.1, 0.9], y in [0.1, 0.9]
    
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.01) # Start small
    
    # Generate grid positions
    # 5 columns
    cols = 5
    rows = int(np.ceil(n / cols)) # 6
    
    x_coords = np.linspace(0.1, 0.9, cols)
    y_coords = np.linspace(0.1, 0.9, rows)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                centers[idx, 0] = x_coords[c]
                centers[idx, 1] = y_coords[r]
                idx += 1
    
    # Optimization parameters
    num_iterations = 3000
    growth_factor = 0.0005 # How fast to grow radii per step
    push_strength = 0.5    # Strength of repulsion
    
    for it in range(num_iterations):
        # 1. Grow radii
        # Calculate max possible radius for each circle based on current geometry
        # Then grow slightly towards it
        
        # We can just grow uniformly or individually. 
        # Individual growth allows adapting to local space.
        # But uniform growth is safer to maintain sum.
        # Let's try uniform growth with a small step, then resolve.
        
        # Actually, growing radii uniformly might be slow.
        # Let's compute available space.
        
        # Update radii
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Distance to boundaries
            dist_bound = min(x, 1-x, y, 1-y)
            
            # Distance to neighbors
            dist_neighbors = np.inf
            for j in range(n):
                if i == j: continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                # The constraint is dist >= r_i + r_j
                # So r_i <= dist - r_j
                limit = dist - radii[j]
                if limit < dist_neighbors:
                    dist_neighbors = limit
            
            max_r = min(dist_bound, dist_neighbors)
            
            # Grow towards max_r
            if max_r > r:
                radii[i] += growth_factor * (max_r - r)
            else:
                # If constrained, maybe shrink slightly to allow movement?
                # But we want to maximize sum.
                # If it's stuck, maybe it's blocked.
                pass

        # 2. Resolve overlaps and adjust positions
        # Calculate forces/shifts
        shifts = np.zeros_like(centers)
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    overlap = min_dist - dist
                    # Normalize direction
                    nx = dx / dist
                    ny = dy / dist
                    
                    # Push apart
                    # Move i away from j, j away from i
                    # Split overlap equally or based on radii?
                    # Equal split is simple.
                    move = overlap * 0.5 # * push_strength? 
                    # If we use push_strength < 1, it dampens.
                    # If > 1, it separates more.
                    
                    # To ensure they separate, we need to move at least overlap/2 each.
                    # But boundaries might prevent full movement.
                    # Let's apply a force proportional to overlap.
                    
                    force = overlap # Simple repulsion
                    
                    shifts[i, 0] += nx * force
                    shifts[i, 1] += ny * force
                    shifts[j, 0] -= nx * force
                    shifts[j, 1] -= ny * force
                elif dist < 1e-9:
                    # Avoid division by zero, push randomly
                    shifts[i, 0] += 0.01
                    shifts[i, 1] += 0.01
                    shifts[j, 0] -= 0.01
                    shifts[j, 1] -= 0.01

        # Apply shifts with a step size (damping)
        # Large shifts might cause oscillation.
        damping = 0.5 
        centers += damping * shifts
        
        # 3. Enforce boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Clamp center to keep circle inside
            # x must be in [r, 1-r]
            # y must be in [r, 1-r]
            # But r changes. 
            # We should clamp based on current r.
            
            min_x = r
            max_x = 1.0 - r
            min_y = r
            max_y = 1.0 - r
            
            centers[i, 0] = np.clip(x, min_x, max_x)
            centers[i, 1] = np.clip(y, min_y, max_y)
            
            # Note: Clamping might cause overlaps again, but next iteration will fix.
            
        # 4. Ensure radii don't exceed bounds after clamping
        # If center is clamped, r might be too big for the new position?
        # Actually, the clamp ensures position is valid for current r.
        # But if we increased r before clamping, we might have been invalid.
        # The clamp fixes position, so validity is restored.
        # However, r might still be too large for the corner?
        # The clamp x >= r ensures x-r >= 0.
        # The clamp x <= 1-r ensures x+r <= 1.
        # So it's valid.

    # Final validation and cleaning
    # Check for any remaining overlaps due to numerical errors or clamping issues
    # and slightly shrink if necessary to be safe.
    # But the problem asks to maximize, so we shouldn't shrink unnecessarily.
    # Just ensure strict validity.
    
    # Re-check overlaps and shrink minimally if needed
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            req = radii[i] + radii[j]
            if dist < req:
                # Overlap detected. Reduce radii slightly?
                # Or move? Moving might violate bounds.
                # Reducing radii is safer for validity, but hurts score.
                # Since we ran many iterations, this should be rare.
                # Let's just accept it might be slightly invalid if we can't fix,
                # but the validation function is strict.
                # Let's try to shrink the larger radius?
                # Or just shrink both.
                overlap = req - dist
                if overlap > 1e-12:
                    # Shrink proportional to radius?
                    # Just shrink both by half overlap
                    reduction = overlap / 2.0 + 1e-6
                    radii[i] -= reduction
                    radii[j] -= reduction
                    radii[i] = max(radii[i], 0.0)
                    radii[j] = max(radii[j], 0.0)
    
    # Check boundary constraints again and shrink if needed
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            # Find max valid radius
            valid_r = min(x, 1-x, y, 1-y)
            radii[i] = max(0.0, valid_r)

    # Compute sum
    total_sum = np.sum(radii)
    
    return centers, radii, float(total_sum)