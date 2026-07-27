# sol_000107 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a3c1a30f) state=993a9880 sum of radii=2.299965 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n_circles = 26
    
    # 1. Initialization: Hexagonal Grid Pattern
    # We aim for a density that suggests r ~ 0.102. 
    # Rows: 5, 5, 5, 5, 5, 1.
    # Vertical spacing factor for hexagonal packing is sqrt(3)/2 ~ 0.866
    
    centers = np.zeros((n_circles, 2))
    radii = np.ones(n_circles) * 0.05 # Initial small radius
    
    row_counts = [5, 5, 5, 5, 5, 1]
    idx = 0
    
    # Estimated geometry for placement
    # Width 1.0. 5 circles. Diameter ~ 0.2.
    # x_spacing = 0.2, y_spacing = 0.1732
    
    x_spacing = 0.20
    y_spacing = 0.1732
    
    # Center the grid vertically and horizontally as much as possible
    # 5 rows of height y_spacing. Total height ~ 4 * y_spacing.
    # We have 6 rows.
    
    current_y = 0.15 # Start with a margin
    
    for i, count in enumerate(row_counts):
        # Stagger every other row
        offset_x = x_spacing / 2.0 if i % 2 == 1 else 0.0
        
        # Calculate x positions to center the row
        # Total width of row with 'count' circles: (count-1)*x_spacing
        # But we want them centered in [0, 1]
        # Actually, simpler: distribute them evenly
        if count > 1:
            row_width = (count - 1) * x_spacing
            start_x = (1.0 - row_width) / 2.0
        else:
            start_x = 0.5
            
        for j in range(count):
            if idx < n_circles:
                centers[idx, 0] = start_x + j * x_spacing + offset_x
                centers[idx, 1] = current_y
                idx += 1
        
        current_y += y_spacing

    # 2. Optimization: Force-Directed Relaxation
    # We will iteratively expand radii and repel overlapping circles.
    # This is a heuristic to find a local maximum.
    
    # Parameters for simulation
    dt = 0.01
    repulsion_strength = 5.0
    expansion_rate = 0.005
    wall_repulsion = 10.0
    
    # Run simulation for a fixed number of steps
    for step in range(2000):
        # 1. Expand radii slightly (simulating growth)
        # We expand based on available space, but we handle overlap in step 2
        # A simple approach: expand all radii by a small factor or amount
        # To maximize sum, we want radii as large as possible.
        # We can just try to set r_i = min(dist_to_others/2, dist_to_wall)
        # But doing this simultaneously causes oscillation.
        # Instead, let's use forces.
        
        forces = np.zeros_like(centers)
        current_radii = radii.copy()
        
        # 2. Calculate forces (Repulsion and Wall constraint)
        for i in range(n_circles):
            xi, yi = centers[i]
            ri = current_radii[i]
            
            # Wall forces
            # If touching wall, force inwards.
            # Also, we want to "pull" circles away from walls to allow growth?
            # Actually, if we are just relaxing, repulsion is enough.
            
            # Try to expand radius
            # Max possible radius based on current neighbors and walls
            max_r = min(xi, 1-xi, yi, 1-yi)
            
            for j in range(n_circles):
                if i == j: continue
                dx = xi - centers[j, 0]
                dy = yi - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist == 0: dist = 1e-9
                # Constraint: dist >= ri + rj
                # If dist < ri + rj, we have overlap.
                # But we want to maximize r.
                # The limiting factor for ri is (dist - rj)
                if dist - current_radii[j] < max_r:
                    max_r = dist - current_radii[j]
            
            if max_r < 0: max_r = 0
            
            # Update radius towards max_r
            # This acts as a "target" radius
            target_r = max_r
            
            # We don't update radii directly here to avoid instant chaos.
            # Instead, we can use the gap to generate a "growth" force or just update.
            # Let's update radii with a small learning rate towards target
            radii[i] = radii[i] + 0.1 * (target_r - radii[i])
            
            # If radii increased and now overlap, we need to move centers.
            # But moving centers is complex.
            # Let's stick to a simpler logic:
            # 1. Set radii to optimal for current centers (LP-like step but greedy)
            # 2. Move centers to reduce overlap.
            
        # Re-calculate optimal radii for current centers
        # This is the key step.
        # For each circle, r_i is limited by walls and neighbors.
        # r_i <= dist_to_wall
        # r_i <= dist_to_j - r_j
        # This is a system of linear inequalities: r_i + r_j <= d_ij
        # We can solve this approximately by iterating.
        
        for _ in range(5): # Inner iteration for radii consistency
            for i in range(n_circles):
                # Constraint from walls
                r_wall = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
                
                # Constraint from neighbors
                r_min = r_wall
                for j in range(n_circles):
                    if i == j: continue
                    dist = np.linalg.norm(centers[i] - centers[j])
                    # r_i + r_j <= dist  => r_i <= dist - r_j
                    if dist - radii[j] < r_min:
                        r_min = dist - radii[j]
                
                if r_min < 0: r_min = 0
                # Update radius. 
                # To maximize sum, we want r_i as large as possible.
                # But if we just set r_i = r_min, it might be stable.
                # However, if r_j is small, r_i can be large.
                # If we update sequentially, we might get order dependency.
                # A relaxation approach:
                radii[i] = r_min 

        # Now, if sum of radii is not increasing or we have overlaps (due to numerical issues or logic),
        # we should perturb centers.
        # Check for overlaps
        has_overlap = False
        for i in range(n_circles):
            for j in range(i+1, n_circles):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < radii[i] + radii[j] - 1e-6:
                    has_overlap = True
                    # Repulsion force
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    if dist == 0: dist = 1e-9
                    # Normalize
                    fx = dx / dist
                    fy = dy / dist
                    # Push apart proportional to overlap
                    overlap = (radii[i] + radii[j]) - dist
                    force = overlap * repulsion_strength
                    
                    centers[i, 0] += fx * force * dt
                    centers[i, 1] += fy * force * dt
                    centers[j, 0] -= fx * force * dt
                    centers[j, 1] -= fy * force * dt

        # Wall constraints for centers
        for i in range(n_circles):
            r = radii[i]
            # Center must be at least r away from walls
            centers[i, 0] = np.clip(centers[i, 0], r, 1-r)
            centers[i, 1] = np.clip(centers[i, 1], r, 1-r)

    # Final clean-up: Ensure no overlaps with tight tolerance
    # Re-solve radii for final centers strictly
    # Use a simple iterative solver for the system r_i + r_j <= d_ij, r_i <= wall_dist
    for _ in range(50):
        for i in range(n_circles):
            r_wall = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
            r_neighbor = np.inf
            for j in range(n_circles):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                r_neighbor = min(r_neighbor, dist - radii[j])
            
            new_r = min(r_wall, r_neighbor)
            if new_r < 0: new_r = 0
            radii[i] = new_r

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
