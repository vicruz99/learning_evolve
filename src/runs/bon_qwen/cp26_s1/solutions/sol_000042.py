# sol_000042 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ff99986a) state=f2e6386d sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed simulation with radius growth.
    """
    n = 26
    
    # 1. Initialize centers in a hexagonal pattern
    # Hexagonal packing is denser than square packing.
    # We estimate an initial radius that fits, e.g., 0.08.
    # Then we let the simulation grow them.
    
    init_r = 0.08
    dx = 2 * init_r
    dy = np.sqrt(3) * init_r
    
    centers = np.zeros((n, 2))
    idx = 0
    
    # Generate grid points
    # We fill rows. Even rows start at x=init_r, odd rows at x=init_r + dx/2
    # y coordinates start at init_r, step dy
    
    y = init_r
    row_idx = 0
    
    while idx < n:
        # Determine x start for this row
        if row_idx % 2 == 0:
            x_start = init_r
        else:
            x_start = init_r + dx / 2
        
        x = x_start
        while x + init_r <= 1.0 + 1e-9:
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
            x += dx
        
        y += dy
        row_idx += 1
        
    # If we didn't fill enough (unlikely with this logic, but safe guard)
    # Fill remaining randomly if needed (should not happen for 26 circles with r=0.08)
    while idx < n:
        centers[idx] = [np.random.uniform(init_r, 1-init_r), np.random.uniform(init_r, 1-init_r)]
        idx += 1

    radii = np.full(n, init_r)
    
    # 2. Simulation parameters
    # We will run a simulation where circles grow and repel each other.
    
    # Learning rates and forces
    repulsion_strength = 5.0
    boundary_strength = 10.0
    growth_rate = 0.0001 # Initial growth
    damping = 0.95 # Velocity damping if we used velocity, but here we use direct pos update
    
    # We'll use a variable step size for radius growth, decreasing over time
    current_growth = 0.001
    
    # Total iterations
    max_iter = 3000
    
    # Temporary arrays for forces
    forces = np.zeros((n, 2))
    
    # To speed up, precompute indices? Not needed for N=26.
    
    for step in range(max_iter):
        # Decrease growth rate over time to fine-tune
        if step > max_iter / 2:
            current_growth = 0.00005
        else:
            current_growth = 0.0005

        # Reset forces
        forces[:] = 0.0
        
        # 1. Calculate pairwise repulsive forces
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                
                # Avoid division by zero
                if dist < 1e-9:
                    dist = 1e-9
                    diff = np.array([1.0, 0.0])
                
                # Overlap amount
                overlap = radii[i] + radii[j] - dist
                
                if overlap > 0:
                    # Repulsive force vector
                    # Push i away from j, j away from i
                    force_vec = diff / dist * overlap * repulsion_strength
                    forces[i] += force_vec
                    forces[j] -= force_vec
        
        # 2. Calculate boundary forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += boundary_strength * (r - x)
            # Right wall
            if x + r > 1:
                forces[i, 0] -= boundary_strength * (x + r - 1)
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += boundary_strength * (r - y)
            # Top wall
            if y + r > 1:
                forces[i, 1] -= boundary_strength * (y + r - 1)
                
        # 3. Update centers
        # Simple Euler integration / Gradient descent on energy
        # Step size for position update
        pos_step = 0.1 
        
        # Apply forces
        centers += forces * pos_step
        
        # Clamp centers to stay roughly inside to prevent runaway, 
        # though boundary forces should handle it. 
        # However, r can be large, so centers must be in [r, 1-r].
        # We can clamp to [0, 1] for now, boundary forces push back.
        centers = np.clip(centers, 0, 1)
        
        # 4. Update radii
        # Increase radii slightly. 
        # If a circle is constrained (high overlap force), it might not grow effectively,
        # but the forces will push neighbors away, creating space.
        
        # Heuristic: Grow radius proportional to available space?
        # Simply adding a constant is simplest and works well with repulsion.
        radii += current_growth
        
        # Cap radii at 0.5 (max possible in unit square)
        radii = np.clip(radii, 0, 0.5)
        
        # Optional: If a circle is completely isolated and far from boundary,
        # it might grow too fast. But repulsion handles neighbors.
        # Boundary handles walls.
        
    # 3. Final cleanup / refinement
    # Sometimes simulation leaves tiny overlaps or boundary violations due to discrete steps.
    # We can do a few aggressive overlap resolution steps without growing radii.
    
    for _ in range(100):
        forces[:] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < 1e-9:
                    dist = 1e-9
                    diff = np.array([1.0, 0.0])
                
                overlap = radii[i] + radii[j] - dist
                if overlap > 0:
                    force_vec = diff / dist * overlap * 10.0 # Strong repulsion
                    forces[i] += force_vec
                    forces[j] -= force_vec
            
            x, y = centers[i]
            r = radii[i]
            if x - r < 0: forces[i, 0] += 10.0 * (r - x)
            if x + r > 1: forces[i, 0] -= 10.0 * (x + r - 1)
            if y - r < 0: forces[i, 1] += 10.0 * (r - y)
            if y + r > 1: forces[i, 1] -= 10.0 * (y + r - 1)
        
        centers += forces * 0.05
        centers = np.clip(centers, 0, 1)

    # 4. Validate and adjust if necessary (sanity check)
    # If any circle is slightly outside, shrink it to fit.
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Check boundaries
        min_dist_to_wall = min(x, 1-x, y, 1-y)
        if r > min_dist_to_wall:
            radii[i] = max(0, min_dist_to_wall) # Shrink radius to fit
            # Recalculate center if needed? 
            # Actually if r was too big, center might be invalid.
            # Let's clamp center to valid range for new r
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1-radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1-radii[i])

    # Check pairwise overlaps and shrink if needed
    # This is a fallback if simulation failed to resolve perfectly
    # We do this by iteratively shrinking the larger circle of an overlapping pair
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                sum_r = radii[i] + radii[j]
                if dist < sum_r - 1e-12:
                    # Overlap detected. Reduce radii to just touch.
                    # Distribute reduction equally or by size?
                    # Let's reduce both equally to maintain balance
                    reduction = (sum_r - dist) / 2
                    radii[i] -= reduction
                    radii[j] -= reduction
                    # Ensure non-negative
                    if radii[i] < 0: radii[i] = 0
                    if radii[j] < 0: radii[j] = 0
                    changed = True
                    
                    # Also ensure centers are consistent (though dist is fixed)
                    # If radii became 0, they fit anywhere.
                    
    # Final validation of boundary again after shrinking
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Re-clamp center if radius shrunk significantly? 
        # Actually if r shrunk, center might be valid now.
        # But if r was shrunk due to boundary, we fixed it.
        # If r shrunk due to overlap, center is fine.
        
        # Just ensure center is within [r, 1-r]
        centers[i, 0] = np.clip(centers[i, 0], r, 1-r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1-r)

    sum_radii = np.sum(radii)
    
    return centers, radii, float(sum_radii)
