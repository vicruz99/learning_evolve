# sol_000256 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3be09fa9) state=677a3823 sum of radii=2.080000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a repulsion-based simulation starting from a hexagonal grid configuration.
    """
    n = 26
    max_iter = 2000
    initial_r = 0.05
    growth_step = 0.0005
    repulsion_strength = 1.0
    
    # 1. Initialize centers using a perturbed hexagonal grid
    # We want to fit 26 circles. A pattern like 5-5-5-5-6 or similar might work, 
    # but a standard hex lattice row arrangement is safer.
    # Let's try to arrange them in rows. 
    # 6 rows with counts: 4, 5, 4, 5, 4, 4 (Total 26) or 5, 4, 5, 4, 5, 3?
    # Let's just generate a dense hex grid and pick the first 26 valid points inside [0,1]x[0,1]
    
    centers = np.zeros((n, 2))
    r = initial_r
    
    # Generate hex grid points
    # Spacing d approx 0.2 (since 1/5 = 0.2)
    d = 0.22 # Slightly larger to fit more
    points = []
    
    # Iterate over a grid to find points
    # Hex grid: y increments by d * sqrt(3)/2. x increments by d. Odd rows shifted by d/2.
    row_height = d * math.sqrt(3) / 2
    
    y = r
    row_idx = 0
    while y + r <= 1.0:
        x = r
        shift = (d / 2) if (row_idx % 2 == 1) else 0
        x += shift
        
        while x + r <= 1.0:
            points.append([x, y])
            x += d
            if len(points) >= n:
                break
        y += row_height
        row_idx += 1
        if len(points) >= n:
            break
            
    # If we didn't get 26 points (unlikely with d=0.22), fallback to random or adjust
    if len(points) < n:
        # Fallback: Random dense packing
        centers = np.random.uniform(0.1, 0.9, (n, 2))
    else:
        # Take the first n points
        centers = np.array(points[:n])

    # 2. Simulation Loop
    # We will try to expand radii and repel circles to maintain validity
    # We keep a single radius for all circles for the simulation to simplify, 
    # but we can allow them to vary if needed. 
    # Actually, maximizing sum of radii often leads to equal radii in dense packings.
    # Let's simulate with equal radii first.
    
    current_r = initial_r
    
    # Precompute neighbor indices for efficiency? Not strictly necessary for N=26
    
    for step in range(max_iter):
        # Try to increase radius
        current_r += growth_step
        
        # Check constraints and compute overlaps
        # We need to find a valid center configuration for this radius
        # If not valid, apply forces
        
        # 1. Boundary checks
        # 2. Overlap checks
        
        # To make it stable, we can perform a relaxation step
        # If we just increased r, circles might overlap. We push them apart.
        
        valid = True
        forces = np.zeros((n, 2))
        
        # Boundary forces
        for i in range(n):
            x, y = centers[i]
            # Left
            if x - current_r < 0:
                forces[i, 0] += (0 - (x - current_r)) * 10.0
            # Right
            if x + current_r > 1:
                forces[i, 0] -= ((x + current_r) - 1) * 10.0
            # Bottom
            if y - current_r < 0:
                forces[i, 1] += (0 - (y - current_r)) * 10.0
            # Top
            if y + current_r > 1:
                forces[i, 1] -= ((y + current_r) - 1) * 10.0
        
        # Overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                dist = math.sqrt(dist_sq)
                
                min_dist = 2 * current_r
                if dist < min_dist:
                    # Overlap amount
                    overlap = min_dist - dist
                    if dist > 1e-9:
                        fx = (dx / dist) * overlap * 5.0
                        fy = (dy / dist) * overlap * 5.0
                    else:
                        # Push apart randomly if centers coincide
                        fx = np.random.uniform(-1, 1)
                        fy = np.random.uniform(-1, 1)
                    
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
                    valid = False

        # Apply forces (damped)
        if not valid:
            # If we are overlapping significantly, we might need to step back or just push hard
            # For stability, let's just apply forces and maybe reduce r slightly if it's too unstable?
            # But we want to maximize r. So we keep r and just move centers.
            # However, if centers hit boundaries and can't move, we might be stuck.
            
            # Limit movement to prevent oscillation
            max_move = 0.005
            for i in range(n):
                f_mag = math.sqrt(forces[i, 0]**2 + forces[i, 1]**2)
                if f_mag > 0:
                    move_x = (forces[i, 0] / f_mag) * min(f_mag * 0.01, max_move)
                    move_y = (forces[i, 1] / f_mag) * min(f_mag * 0.01, max_move)
                    
                    centers[i, 0] += move_x
                    centers[i, 1] += move_y
                    
                    # Clamp to bounds
                    centers[i, 0] = np.clip(centers[i, 0], 0, 1)
                    centers[i, 1] = np.clip(centers[i, 1], 0, 1)
        
        # If valid for a while, we might want to accelerate growth?
        # Or just keep growing slowly.
        
        # Check if we are stuck (forces are zero or small but overlaps persist due to boundary)
        # If overlaps persist and we can't move, we should decrease r.
        if not valid:
            # Check if any movement was possible
            # If forces are huge but positions didn't change (clamped), we are stuck.
            # Heuristic: if many overlaps, shrink r slightly
            overlap_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < 2 * current_r - 1e-6:
                        overlap_count += 1
            
            if overlap_count > n: # Severe overlap
                 current_r -= growth_step * 5 # Retract
                 # Reset forces?
                 forces = np.zeros((n, 2))
        
    # 3. Final Refinement
    # After simulation, we have a configuration with radius 'current_r'.
    # We can try to optimize individually or just accept.
    # To be safe and precise, let's run a few steps of coordinate descent to maximize sum of radii
    # allowing radii to vary slightly if it helps, but equal is likely good.
    # Let's just return the equal radius solution found.
    
    radii = np.ones(n) * current_r
    
    # Clean up: ensure strict validity within tolerance
    # Sometimes floating point errors might cause tiny overlaps.
    # Let's verify and fix if needed.
    
    # Re-check and adjust radii if needed to be strictly valid
    # The validation function allows 1e-12 tolerance.
    
    # Let's try to slightly reduce radii if any tiny overlap exists to be safe
    # But we want to maximize sum, so we want them as large as possible.
    
    # Final validation check logic (local)
    for i in range(n):
        x, y = centers[i]
        r_i = radii[i]
        # Boundary
        r_i = min(r_i, x, 1-x, y, 1-y)
        radii[i] = r_i
        
    # Check pairwise
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            r_sum = radii[i] + radii[j]
            if dist < r_sum:
                # Overlap, reduce radii equally
                reduction = (r_sum - dist) / 2 + 1e-7
                radii[i] -= reduction
                radii[j] -= reduction
    
    # Ensure non-negative
    radii = np.maximum(radii, 0.0)
    
    # Recalculate sum
    sum_radii = np.sum(radii)
    
    # The simulation might have settled at a lower radius if stuck.
    # Let's try a quick random restart if sum is low (e.g. < 2.4)
    # But for 26 circles, > 2.4 is expected.
    
    return centers, radii, sum_radii
