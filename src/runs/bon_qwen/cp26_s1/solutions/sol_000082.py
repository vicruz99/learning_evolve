# sol_000082 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b0810f40) state=a0be5760 sum of radii=0.025828 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square [0,1]x[0,1]
    to maximize the sum of radii.
    """
    np.random.seed(42)
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # Distribute circles into 6 rows: 5, 5, 5, 5, 5, 1
    # This shape fits the square better than a 5x5 grid.
    rows = [5, 5, 5, 5, 5, 1]
    
    # Initial radius estimation based on hexagonal packing density
    # Height required ~ 2r + 5*(r*sqrt(3)) <= 1  =>  r <= 1 / (2 + 5*sqrt(3))
    # We start with a slightly smaller radius to ensure valid start
    r_target = 0.09
    h = r_target * math.sqrt(3)
    
    centers = np.zeros((n, 2))
    radii = np.full(n, r_target)
    
    idx = 0
    for i, count in enumerate(rows):
        # Horizontal spacing 2r, centered horizontally in the square
        row_width = (count - 1) * 2 * r_target
        x_start = (1.0 - row_width) / 2.0
        
        # Shift odd rows (0-indexed) by 1 radius to nest them
        if i % 2 == 1:
            x_start += r_target
        
        for j in range(count):
            centers[idx, 0] = x_start + j * 2 * r_target
            centers[idx, 1] = r_target + i * h
            idx += 1
            
    # 2. Force-Directed Optimization
    # Iteratively expand radii while repelling overlaps
    
    max_iterations = 2000
    step_size = 0.05 # Strength of repulsion
    repel_scale = 5.0 # Multiplier for overlap repulsion
    
    # Current radii
    current_r = 0.09
    
    # Pre-calculate row structure to apply tension
    # Tension helps keep the packing compact and central
    tension_strength = 0.001 
    
    for iteration in range(max_iterations):
        forces = np.zeros((n, 2))
        overlap_count = 0
        
        # A. Compute Repulsive Forces between circles
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = math.hypot(dx, dy)
                
                r_sum = radii[i] + radii[j]
                
                if dist < r_sum:
                    # Overlap detected: strong repulsion
                    overlap_count += 1
                    # Force proportional to overlap amount, inversely to distance
                    if dist > 1e-9:
                        # Normalize vector
                        fx = (dx / dist) * (r_sum - dist) * repel_scale
                        fy = (dy / dist) * (r_sum - dist) * repel_scale
                    else:
                        # If on top of each other, push in random direction
                        fx = np.random.normal()
                        fy = np.random.normal()
                    
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
                else:
                    # Soft repulsion to prevent sticking if they get too close
                    if dist < 1.5 * r_sum:
                        dist_factor = 1.0 / (dist * dist + 1e-9)
                        fx = (dx / dist) * dist_factor * 0.1
                        fy = (dy / dist) * dist_factor * 0.1
                        
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy

        # B. Boundary Constraints (Repulsion from walls)
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x < r:
                forces[i, 0] += (r - x) * repel_scale
            # Right wall
            if x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * repel_scale
            # Bottom wall
            if y < r:
                forces[i, 1] += (r - y) * repel_scale
            # Top wall
            if y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * repel_scale

        # C. Center Tension (Keeps circles packed together)
        # Pull centers towards (0.5, 0.5) slightly
        for i in range(n):
            forces[i, 0] += (0.5 - centers[i, 0]) * tension_strength
            forces[i, 1] += (0.5 - centers[i, 1]) * tension_strength

        # D. Update Positions
        # Adaptive step size: reduce if many overlaps to prevent instability
        if overlap_count > 0:
            current_step = step_size * (0.1 / (overlap_count + 1))
        else:
            current_step = step_size
            
        centers += forces * current_step

        # Clip positions strictly to [0, 1] to ensure validity during process
        centers = np.clip(centers, 0, 1)

        # E. Try to Increase Radius
        # If no overlaps, slowly grow the circles
        if overlap_count == 0:
            growth = 0.0001
            radii += growth
        else:
            # If overlapping, shrink slightly to recover
            radii -= 0.0001
            radii = np.maximum(radii, 0.001)

    # 3. Final Cleanup
    # Ensure strict validity
    # Clip any slight boundary violations caused by float errors
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Hard clip centers if needed, though forces should handle it
        if x - r < 0: centers[i, 0] = r
        if x + r > 1: centers[i, 0] = 1 - r
        if y - r < 0: centers[i, 1] = r
        if y + r > 1: centers[i, 1] = 1 - r
        
        # Safety check for overlaps (rarely needed after forces, but good for robustness)
        # If an overlap persists, reduce the radius of the pair slightly
        # (Simple post-processing)
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist = math.hypot(dx, dy)
            r_sum = radii[i] + radii[j]
            if dist < r_sum - 1e-9:
                # Scale down radii to fit distance
                scale = dist / r_sum
                radii[i] *= scale
                radii[j] *= scale
                # Re-check boundaries after scaling
                radii[i] = min(radii[i], centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
                radii[j] = min(radii[j], centers[j,0], 1-centers[j,0], centers[j,1], 1-centers[j,1])

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
