# sol_000353 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b8d6b6a1) state=e1c5f450 sum of radii=1.544324 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    # Constants for simulation
    max_iterations = 2000
    initial_radius = 0.01
    radius_step = 0.0002
    force_strength = 1.0
    damping = 0.5
    
    # Initialize centers using a hexagonal lattice pattern
    # We estimate a spacing that fits roughly 5-6 rows
    centers = np.zeros((n, 2))
    
    # Let's try to fit them in a grid first, then perturb to hexagonal
    # A 5x5 grid has 25 points. We have 26.
    # Let's create a denser grid and pick points, or just construct rows.
    
    # Hexagonal packing construction
    # We want to distribute 26 points. 
    # Approximate rows needed: sqrt(26) ~ 5.1. Let's use 6 rows.
    # Row counts: 5, 4, 5, 4, 5, 3? Sum = 26.
    # Or 5, 5, 5, 5, 6 (5 rows).
    
    # Let's use 6 rows for better vertical distribution
    row_counts = [5, 4, 5, 4, 5, 3] # Sum = 26
    # Adjust row heights to fit in [0, 1]
    # We will place centers roughly at y = (row_index + 0.5) * (1.0 / num_rows)
    # But hexagonal spacing is better.
    
    # Let's just generate a dense hexagonal grid and take the first 26 points
    # that fall within a slightly shrunk square to allow boundary breathing room
    # Then optimize.
    
    pts = []
    # Grid parameters
    dx = 0.25
    dy = dx * np.sqrt(3) / 2
    
    # Generate points
    y = dy / 2 # Start offset
    row = 0
    while y < 1.0:
        x = 0.0
        # Offset odd rows
        offset = (dx / 2) if row % 2 == 1 else 0.0
        x = offset
        while x < 1.0:
            pts.append([x, y])
            x += dx
        y += dy
        row += 1
    
    # If we have more than 26, take a subset that is well distributed
    # Or just take first 26
    if len(pts) >= n:
        # Take points that are somewhat central to avoid boundary issues initially?
        # Actually, taking first 26 from a grid starting at 0,0 is fine, optimizer will move them.
        # But starting at 0,0 implies radius 0.
        # Let's shift them inside.
        centers = np.array(pts[:n])
        # Add small random noise to break symmetry
        centers += np.random.rand(n, 2) * 0.01
    else:
        # Fallback to random if grid didn't generate enough (unlikely)
        centers = np.random.rand(n, 2) * 0.8 + 0.1

    # Ensure initial centers are strictly inside (0,1)
    centers = np.clip(centers, 0.05, 0.95)

    # Simulation loop to inflate radii
    current_radius = initial_radius
    
    # We will perform the optimization by trying to increase radius
    # and resolving collisions.
    
    # Precompute indices for pairs to speed up loop
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for _ in range(max_iterations):
        # Try to increase radius
        # We check if we can increase radius by a small amount without immediate overlap
        # But simpler: just apply forces for the current radius.
        # If the system is in equilibrium with current radius, we try to increase radius.
        
        # Actually, a robust method is:
        # 1. Apply forces to resolve overlaps for current_radius.
        # 2. If max displacement is small, increase radius.
        
        moves = np.zeros_like(centers)
        
        # Check inter-circle overlaps
        for i, j in pair_indices:
            diff = centers[i] - centers[j]
            dist = np.linalg.norm(diff)
            min_dist = 2 * current_radius
            
            if dist < min_dist:
                overlap = min_dist - dist
                # Normalize direction
                if dist > 1e-9:
                    direction = diff / dist
                else:
                    direction = np.random.rand(2) * 2 - 1
                    direction /= np.linalg.norm(direction)
                
                # Push apart
                push = overlap / 2.0
                moves[i] += direction * push
                moves[j] -= direction * push
        
        # Check boundary overlaps
        for i in range(n):
            x, y = centers[i]
            r = current_radius
            
            # Left wall
            if x < r:
                moves[i, 0] += (r - x)
            # Right wall
            if x > 1 - r:
                moves[i, 0] -= (x - (1 - r))
            # Bottom wall
            if y < r:
                moves[i, 1] += (r - y)
            # Top wall
            if y > 1 - r:
                moves[i, 1] -= (y - (1 - r))

        # Apply moves
        # Limit move size to prevent instability? 
        # The forces are proportional to overlap, so it should be stable.
        centers += moves
        
        # Clip to [0, 1] just in case (though logic above should keep it close)
        # But strictly, center must be in [r, 1-r].
        # If we clip to [0,1], we might violate radius constraint if r is large.
        # But the forces push back from boundaries, so centers should stay within [r, 1-r]
        # if the simulation converges.
        
        # Check for convergence or stability to increase radius
        # If max overlap is small, we can try to increase radius
        max_overlap = 0.0
        for i, j in pair_indices:
            dist = np.linalg.norm(centers[i] - centers[j])
            overlap = 2 * current_radius - dist
            if overlap > max_overlap:
                max_overlap = overlap
        
        for i in range(n):
            x, y = centers[i]
            r = current_radius
            # Boundary overlaps
            bl = max(0, r - x)
            br = max(0, x - (1 - r))
            bd = max(0, r - y)
            bt = max(0, y - (1 - r))
            max_overlap = max(max_overlap, bl, br, bd, bt)
            
        if max_overlap < 1e-6:
            # Stable, try to increase radius
            # Adaptive step?
            current_radius += radius_step
            # If radius gets too large, it will fail to resolve, loop continues
            # We might want to stop if radius is too large to be feasible?
            # No, the loop runs fixed iterations.
            # But if radius keeps increasing and overlaps become huge, it might oscillate.
            # A check: if radius > 0.5, break? (Max possible is 0.5)
            if current_radius >= 0.5:
                current_radius = 0.5
                break
        else:
            # If overlaps are large, we shouldn't increase radius.
            # We just continue resolving overlaps for current radius.
            # Maybe reduce step if we are stuck?
            pass

    # After simulation, verify and compute final radius
    # The radius might be slightly optimistic due to discrete steps.
    # We should compute the exact max radius supported by the final configuration.
    
    # Calculate max feasible radius for the current centers
    # r_max = min( min_pair_dist / 2, min_boundary_dist )
    
    min_pair_dist = float('inf')
    for i, j in pair_indices:
        d = np.linalg.norm(centers[i] - centers[j])
        if d < min_pair_dist:
            min_pair_dist = d
            
    min_boundary_dist = float('inf')
    for i in range(n):
        x, y = centers[i]
        d = min(x, 1 - x, y, 1 - y)
        if d < min_boundary_dist:
            min_boundary_dist = d
            
    r_final = min(min_pair_dist / 2, min_boundary_dist)
    
    # To be safe against numerical errors, subtract a tiny epsilon
    r_final = max(0.0, r_final - 1e-9)
    
    # Create radii array
    radii = np.full(n, r_final)
    sum_radii = np.sum(radii)
    
    # One final check and adjustment if any circle is invalid
    # Although the logic above should ensure validity.
    # Just in case, enforce constraints strictly.
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Boundary fix
        if x < r: x = r
        if x > 1 - r: x = 1 - r
        if y < r: y = r
        if y > 1 - r: y = 1 - r
        centers[i] = [x, y]

    return centers, radii, sum_radii
