# sol_000199 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 263f0241) state=5e3900f5 sum of radii=1.035955 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Optimizes the positions and radii of 26 circles in a unit square 
    to maximize the sum of radii.
    """
    n = 26
    num_iterations = 2000
    initial_radius = 0.05
    growth_rate = 1e-4
    force_strength = 0.5
    damping = 0.9
    
    # Initialize centers in a hexagonal grid pattern
    centers = np.zeros((n, 2))
    
    # Hexagonal packing parameters
    # We try to fit points in a hexagonal lattice
    # Rows
    rows = 6
    # Approximate spacing
    # We will refine positions via optimization
    
    # Generate a hexagonal grid of points
    points = []
    r_init = 0.06 # Initial spacing estimate
    dy = r_init * math.sqrt(3)
    dx = r_init * 2.0
    
    row_idx = 0
    count = 0
    y = r_init
    
    while count < n:
        x = r_init
        # Offset for odd/even rows
        if row_idx % 2 == 1:
            x = r_init + dx / 2.0
        
        while x <= 1.0 - r_init:
            points.append([x, y])
            count += 1
            if count == n:
                break
            x += dx
        
        y += dy
        row_idx += 1
    
    # If we didn't get enough points (unlikely with dense grid), fill with random
    while len(points) < n:
        points.append([np.random.rand(), np.random.rand()])
    
    centers = np.array(points[:n])
    
    # Initialize radii
    radii = np.full(n, initial_radius)
    
    # Optimization loop
    # We will iteratively try to expand radii and resolve collisions
    # This acts like a simulated annealing or force-directed layout
    
    # Precompute indices for efficiency
    i_indices, j_indices = np.triu_indices(n, k=1)
    
    for step in range(num_iterations):
        # Increase radii
        # Adaptive growth: smaller as we progress
        current_growth = growth_rate * (1.0 - 0.5 * (step / num_iterations))
        radii += current_growth
        
        # Calculate overlaps and forces
        # Forces to push centers apart
        forces = np.zeros_like(centers)
        
        # Check boundary constraints
        # If a circle is too close to boundary, push it in
        # Effective radius r_i. Constraint: x >= r_i, x <= 1-r_i
        # Violation amount
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += force_strength * (r - x)
            # Right wall
            if x + r > 1:
                forces[i, 0] -= force_strength * (x + r - 1)
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += force_strength * (r - y)
            # Top wall
            if y + r > 1:
                forces[i, 1] -= force_strength * (y + r - 1)

        # Check inter-circle constraints
        # dist >= r_i + r_j
        # If dist < r_i + r_j, overlap depth = (r_i + r_j) - dist
        # Push apart along vector
        
        # Vectorized distance calculation
        # centers shape (n, 2)
        # diff shape (M, 2) where M is number of pairs
        
        diffs = centers[i_indices] - centers[j_indices]
        dists = np.linalg.norm(diffs, axis=1)
        sums_r = radii[i_indices] + radii[j_indices]
        
        overlaps = sums_r - dists
        overlaps[overlaps < 0] = 0 # Only care about positive overlaps
        
        # Compute force vectors
        # Force is proportional to overlap, direction is unit vector * diff
        # Avoid division by zero
        safe_dists = np.maximum(dists, 1e-9)
        unit_vecs = diffs / safe_dists[:, np.newaxis]
        
        # Force magnitude
        f_mags = force_strength * overlaps
        
        # Apply forces
        # Accumulate forces for each circle
        np.add.at(forces, i_indices, f_mags[:, np.newaxis] * unit_vecs)
        np.add.at(forces, j_indices, -f_mags[:, np.newaxis] * unit_vecs)
        
        # Update centers
        # Use a step size for movement
        move_step = 0.5 * (1.0 - 0.8 * (step / num_iterations))
        centers += move_step * forces
        
        # Damping forces for next step? No, forces recomputed.
        # But we can dampen the movement or forces.
        
        # Clamp centers to valid range to prevent wild moves
        # Although forces should handle it, numerical stability
        centers = np.clip(centers, 0.0, 1.0)
        
        # Optional: If radii are growing too fast and causing instability, reduce step
        # But the damping in move_step helps.

    # Final cleanup: Ensure valid radii (non-negative)
    radii = np.maximum(radii, 1e-9)
    
    # Validate and adjust if necessary (simple projection)
    # If a circle is outside, shrink it to fit
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Shrink to fit boundary
        max_r_x = min(x, 1-x)
        max_r_y = min(y, 1-y)
        max_r = min(max_r_x, max_r_y)
        if radii[i] > max_r:
            radii[i] = max_r
            
    # Check overlaps one last time and resolve by shrinking smaller circle?
    # Or just return. The optimization should have resolved major overlaps.
    # However, to be safe for the validator:
    # If overlaps exist, we might fail validation.
    # Let's do a quick local shrinking pass to guarantee validity.
    
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-9:
                    # Overlap detected
                    # Reduce radii slightly to resolve
                    # Reduce both proportionally to their size or just halve overlap
                    overlap = (radii[i] + radii[j]) - dist
                    reduction = overlap / 2.0
                    radii[i] -= reduction
                    radii[j] -= reduction
                    radii[i] = max(0, radii[i])
                    radii[j] = max(0, radii[j])
                    changed = True
                    # Re-check bounds
                    for k in range(2):
                         # Re-check bounds for these two
                         xc, yc = centers[k] # wait k is index
                         # actually just re-run the loop or check
                         pass 
                    # Since we reduced radii, boundary is still satisfied if it was before.
                    break 
            if changed: break

    sum_radii = np.sum(radii)
    return centers, radii, float(sum_radii)

# Run the function to get the result
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
