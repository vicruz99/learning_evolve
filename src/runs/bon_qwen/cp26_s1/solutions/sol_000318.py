# sol_000318 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 78c934c9) state=df156f53 sum of radii=2.426666 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    # 1. Initial Configuration (Hexagonal Lattice)
    n = 26
    centers = np.zeros((n, 2))
    
    # Generate a hexagonal lattice
    # We start with a slightly larger grid to ensure we have enough points
    pts = []
    # Approximate number of circles per side for hexagonal packing
    # density is higher, so we can estimate fewer rows/cols than square grid
    # 26 circles -> approx 6x5 in a grid, but hex is tighter. 
    # Let's generate a 10x10 lattice and trim.
    for i in range(12):
        for j in range(12):
            # Hex spacing: x spacing 1, y spacing sqrt(3)/2
            # Shift even rows by 0.5
            x = i + (j % 2) * 0.5
            y = j * math.sqrt(3) / 2
            pts.append([x, y])
    
    pts = np.array(pts)
    
    # Normalize to unit square roughly and select 26 points
    # Find min/max to center and scale
    if pts.shape[0] > 0:
        min_pt = pts.min(axis=0)
        max_pt = pts.max(axis=0)
        pts = (pts - min_pt) / (max_pt - min_pt) * 0.8 + 0.1 # Fit in [0.1, 0.9]
        
        # Select 26 points that are most spread out (simple heuristic: take from list)
        # Or just take the first 26 if the grid was dense enough
        if pts.shape[0] > n:
            # Sort by distance from center to pick outer ones first? 
            # Or just take first n. 
            # To maximize sum, we want them evenly spread. 
            # Let's just take the first n from the dense grid.
            centers[:n] = pts[:n]
        else:
            # If not enough points, add random ones or extend
            # (Unlikely with 12x12 grid)
            pass
            
    # 2. Iterative Optimization (Force Directed)
    # Initialize radii based on clearance
    radii = np.zeros(n)
    for i in range(n):
        r = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                r = min(r, d/2)
        radii[i] = r

    # Force simulation parameters
    iterations = 3000
    alpha = 0.99 # Decay for step size / temperature
    step_size = 0.005
    
    for k in range(iterations):
        forces = np.zeros((n, 2))
        current_step = step_size * (alpha ** (k / 1000.0))
        
        # Calculate forces and update radii
        # We simulate repulsion when circles overlap or are too close relative to their potential size
        # However, we want to maximize sum of radii. 
        # Heuristic: If a circle is "tight" (radius limited by neighbor/boundary), push it away.
        
        # First, update radii to their maximum possible for current centers
        for i in range(n):
            max_r = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(centers[i] - centers[j])
                    max_r = min(max_r, d/2) # Assuming equal radii for simple repulsion logic
            radii[i] = max_r

        sum_radii = np.sum(radii)
        
        # Apply repulsive forces based on "pressure"
        # If circles are touching, they repel.
        # We use a soft potential to guide them apart.
        for i in range(n):
            # Force from boundaries
            if centers[i,0] - radii[i] < 1e-4: forces[i, 0] += 1.0
            if centers[i,0] + radii[i] > 1 - 1e-4: forces[i, 0] -= 1.0
            if centers[i,1] - radii[i] < 1e-4: forces[i, 1] += 1.0
            if centers[i,1] + radii[i] > 1 - 1e-4: forces[i, 1] -= 1.0
            
            for j in range(i + 1, n):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # If circles are touching or overlapping
                if dist < radii[i] + radii[j] + 0.01: 
                    # Repulsive force proportional to overlap/magnitude
                    # Direction is along the vector connecting centers
                    if dist > 1e-9:
                        fx = dx / dist
                        fy = dy / dist
                        # Strength of force: stronger if closer
                        # We want to separate them to increase sum of radii
                        # A simple inverse square or constant push
                        strength = 1.0 / (dist + 1e-4) 
                        forces[i, 0] += fx * strength
                        forces[i, 1] += fy * strength
                        forces[j, 0] -= fx * strength
                        forces[j, 1] -= fy * strength

        # Update centers
        for i in range(n):
            # Normalize force to avoid huge jumps
            f_norm = np.linalg.norm(forces[i])
            if f_norm > 0:
                centers[i] += (forces[i] / f_norm) * current_step
            
            # Clamp to valid region [0,1] with a small margin
            centers[i,0] = np.clip(centers[i,0], 0.001, 0.999)
            centers[i,1] = np.clip(centers[i,1], 0.001, 0.999)

    # 3. Final Radius Calculation
    # After optimization, calculate the exact maximum radii for the final centers
    for i in range(n):
        r = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                r = min(r, d/2)
        radii[i] = r

    # Final validation and correction (just in case of numerical drift)
    # If any overlap, reduce radii slightly
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < radii[i] + radii[j]:
                # Scale down to fit
                scale = d / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale
    
    # Ensure non-negative and valid bounds
    radii = np.maximum(radii, 0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
