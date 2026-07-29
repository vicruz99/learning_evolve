# sol_000074 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ed1177e6) state=8974d275 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    centers = np.zeros((n, 2))
    
    # 1. Initialize on a Hexagonal Lattice
    # Hexagonal packing is denser. We try to fit rows.
    # Estimation: r ~ 0.1. Vertical spacing ~ sqrt(3)*r ~ 0.173.
    # Height 1.0 fits about 6-7 rows.
    # We distribute points in a staggered grid pattern.
    
    # Heuristic initialization:
    # Try to place points in a grid that approximates hexagonal packing.
    # Number of rows approx sqrt(N * sqrt(3)) ~ 6-7.
    num_rows = 7
    row_counts = [4, 5, 4, 5, 4, 5, 4] # Sum = 27? 4+5+4+5+4+5+4 = 31. Too many.
    # Let's try fewer rows or fewer per row.
    # N=26. 
    # Pattern 5, 4, 5, 4, 5, 4, 3? Sum = 30.
    # Pattern 5, 4, 5, 4, 5, 3? Sum = 26. (6 rows)
    
    rows_pattern = [5, 4, 5, 4, 5, 3]
    current_idx = 0
    
    # Vertical spacing estimate
    y_spacing = 1.0 / (len(rows_pattern) + 1) * 1.5 # Rough guess to spread them
    
    # Better initialization: Random or structured?
    # Let's use a structured hex-like grid.
    # We will refine this with optimization anyway.
    
    # Let's place them in a distorted grid first
    # 6 rows
    y_coords = np.linspace(0.1, 0.9, len(rows_pattern))
    
    k = 0
    for i, count in enumerate(rows_pattern):
        y = y_coords[i]
        # x coordinates centered
        # width available ~ 0.8. 
        # spacing ~ 0.8 / (count + 1) ?
        # Just spread them evenly
        x_coords = np.linspace(0.1, 0.9, count)
        
        # Stagger odd rows?
        if i % 2 == 1:
            x_coords = x_coords + (0.9 - 0.1) / (2 * count)
        
        for j in range(count):
            if k < n:
                centers[k] = [x_coords[j], y]
                k += 1
                
    # Fill remaining if any (should be exact 26)
    if k < n:
        # Fallback random placement for any leftovers
        np.random.seed(42)
        while k < n:
            centers[k] = [np.random.uniform(0.05, 0.95), np.random.uniform(0.05, 0.95)]
            k += 1

    # 2. Optimization Loop (Repulsive Forces)
    # We want to maximize the minimum distance between circles and boundaries.
    # This is equivalent to minimizing energy E = sum(1/dist^2) + sum(1/dist_wall^2)
    
    # Initial step size
    step_size = 0.05
    decay = 0.995
    max_iter = 2000
    
    # To avoid division by zero, add small epsilon to distances
    eps = 1e-6
    
    for _ in range(max_iter):
        forces = np.zeros_like(centers)
        
        # Inter-circle repulsion
        # O(N^2) is fine for N=26
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < eps:
                    dist = eps
                    # Random nudge
                    diff = np.random.rand(2) - 0.5
                    dist = np.linalg.norm(diff)
                
                # Force magnitude ~ 1/dist^2 (Coulombic)
                # Direction: push apart
                force_mag = 1.0 / (dist * dist)
                force_vec = (diff / dist) * force_mag
                
                forces[i] += force_vec
                forces[j] -= force_vec
        
        # Boundary repulsion
        # Push away from x=0, x=1, y=0, y=1
        for i in range(n):
            x, y = centers[i]
            
            # Distance to walls
            dx_left = x
            dx_right = 1.0 - x
            dy_bottom = y
            dy_top = 1.0 - y
            
            # Force from left wall (pushes right)
            if dx_left < 0.5: # Only if close? No, 1/x^2 works everywhere but strong near 0
                # Actually 1/x^2 is strong near 0, weak far. 
                # We want to keep them inside. 
                # Potential 1/x pushes away from 0.
                force_x = 1.0 / (dx_left * dx_left + eps)
            else:
                force_x = 0 # Already safe from left? 
                # But we want to balance. 
                # Better: Force = 1/d^2 always, but directed outwards.
            
            # Refined Boundary Force:
            # F = 1/d^2 directed away from wall.
            
            # Left wall
            if x < 0.5:
                forces[i, 0] += 1.0 / (x * x + eps)
            # Right wall
            if x > 0.5:
                forces[i, 0] -= 1.0 / ((1-x)*(1-x) + eps)
            # Bottom wall
            if y < 0.5:
                forces[i, 1] += 1.0 / (y * y + eps)
            # Top wall
            if y > 0.5:
                forces[i, 1] -= 1.0 / ((1-y)*(1-y) + eps)

        # Update positions
        centers += forces * step_size
        
        # Clamp to bounds (safety)
        centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
        
        step_size *= decay
        
        if step_size < 1e-7:
            break

    # 3. Calculate Radius
    # r is limited by min distance between centers / 2 and min distance to wall
    min_dist = 1.0
    
    # Center-to-center
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
    
    r = min_dist / 2.0
    
    # Center-to-wall
    for i in range(n):
        x, y = centers[i]
        dist_wall = min(x, 1-x, y, 1-y)
        if dist_wall < r:
            r = dist_wall
            
    # 4. Adjust Centers to maximize r if possible
    # If r is determined by wall distance, we might be able to shift centers inwards.
    # However, the force simulation should have balanced this.
    # Just to be safe, if r is very small, we might have converged to a bad local min.
    # But with repulsive forces, it usually finds a good spread.
    
    # Scale centers to fit radius r exactly against the tightest constraint?
    # Actually, if we just output r calculated from positions, it is valid.
    # But we might be able to increase r slightly by shifting?
    # The simulation pushes them apart, so they are likely touching.
    
    # Let's ensure r is not larger than wall distance
    # Recompute r strictly
    r_calc = 1.0
    # Check pairs
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(centers[i] - centers[j]))
    r_calc = min(r_calc, min(dists)/2.0) if dists else 1.0
    
    # Check walls
    for i in range(n):
        dist_w = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        r_calc = min(r_calc, dist_w)
        
    # It's possible the simulation pushed them to boundaries where r is limited by wall.
    # If so, r_calc is correct.
    
    radii = np.full(n, r_calc)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
