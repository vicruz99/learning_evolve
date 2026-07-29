# sol_000023 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d7881bb) state=0e0ac4c8 sum of radii=1.236966 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Uses a force-directed layout with an inflating radius strategy.
    """
    n_circles = 26
    
    # 1. Initialize positions using a hexagonal lattice subset
    # This provides a good dense starting configuration
    pts = []
    # Generate a grid of points
    for row in range(10):
        for col in range(10):
            x = col * 1.0 + (row % 2) * 0.5
            y = row * (math.sqrt(3)/2)
            pts.append([x, y])
    
    # Select the first 26 points (a compact block)
    pts = np.array(pts[:n_circles])
    
    # Center and scale to fit in [0,1] with a margin
    min_p = pts.min(axis=0)
    max_p = pts.max(axis=0)
    size = np.max(max_p - min_p)
    
    if size > 1e-9:
        # Scale to 0.9 size to allow room for expansion
        scale = 0.9 / size
        pts = (pts - min_p) * scale
        # Center in the unit square
        current_center = (pts.max(axis=0) + pts.min(axis=0)) / 2.0
        pts = pts - current_center + 0.5
    else:
        pts = np.random.rand(n_circles, 2)

    centers = pts
    # Start with small radii
    radii = np.ones(n_circles) * 0.02 
    
    # 2. Simulation Parameters
    num_iterations = 8000
    dt = 0.0005
    growth_rate = 0.00002
    repulsion_strength = 20.0 
    boundary_strength = 50.0 
    
    # 3. Main Optimization Loop
    for step in range(num_iterations):
        forces = np.zeros_like(centers)
        max_overlap = 0.0
        
        # Pairwise repulsion forces
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq)
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    overlap = min_dist - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
                    
                    if dist > 1e-9:
                        nx = dx / dist
                        ny = dy / dist
                        f = overlap * repulsion_strength
                        forces[i, 0] += nx * f
                        forces[i, 1] += ny * f
                        forces[j, 0] -= nx * f
                        forces[j, 1] -= ny * f
        
        # Boundary repulsion forces
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x < r:
                ov = r - x
                if ov > max_overlap: max_overlap = ov
                forces[i, 0] += (r - x) * boundary_strength
            # Right wall
            elif x > 1 - r:
                ov = x - (1 - r)
                if ov > max_overlap: max_overlap = ov
                forces[i, 0] -= (x - (1 - r)) * boundary_strength
            
            # Bottom wall
            if y < r:
                ov = r - y
                if ov > max_overlap: max_overlap = ov
                forces[i, 1] += (r - y) * boundary_strength
            # Top wall
            elif y > 1 - r:
                ov = y - (1 - r)
                if ov > max_overlap: max_overlap = ov
                forces[i, 1] -= (y - (1 - r)) * boundary_strength

        # Update positions
        centers += forces * dt
        
        # Clip centers to [0, 1] to prevent escaping
        centers = np.clip(centers, 0.0, 1.0)

        # Adaptive Radius Growth
        # Increase radius if overlaps are small
        if max_overlap < 1e-4:
            radii += growth_rate
        else:
            # If overlap is significant, we might want to reduce radius to escape local minima
            # But usually just waiting is fine.
            pass

        # Cooling schedule to settle
        if step > 5000:
            dt *= 0.9995
            growth_rate *= 0.995

    # 4. Final Correction to ensure strict validity
    # Iteratively shrink radii if overlaps persist
    for _ in range(1000):
        ov = 0.0
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                req = radii[i] + radii[j]
                if dist < req:
                    ov = max(ov, req - dist)
            
            x, y = centers[i]
            r = radii[i]
            if x - r < 0: ov = max(ov, -(x-r))
            if x + r > 1: ov = max(ov, x+r-1)
            if y - r < 0: ov = max(ov, -(y-r))
            if y + r > 1: ov = max(ov, y+r-1)
        
        if ov < 1e-12:
            break
        
        # Shrink radii to resolve overlaps
        radii -= ov * 0.51 
        
    # Ensure non-negative radii
    radii = np.maximum(radii, 0.0)
    
    sum_radii = float(np.sum(radii))
    
    return centers, radii, sum_radii
