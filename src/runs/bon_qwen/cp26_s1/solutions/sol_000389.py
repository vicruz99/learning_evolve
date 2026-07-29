# sol_000389 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 19a68663) state=4318e1f0 sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    
    # 1. Initialize centers and radii
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.05
    
    # Hexagonal packing initialization
    # Rows: 5, 5, 5, 5, 5, 1
    row_counts = [5, 5, 5, 5, 5, 1]
    current_idx = 0
    
    # Vertical spacing for hexagonal packing
    # Using a target radius of ~0.1, spacing is r*sqrt(3)
    row_h = 0.1 * math.sqrt(3) 
    start_y = 0.1
    
    for i, count in enumerate(row_counts):
        y = start_y + i * row_h
        
        # Shift odd rows to form hexagonal pattern
        shift = 0.05 if i % 2 == 1 else 0.0
        start_x = 0.1 + shift
        
        for j in range(count):
            x = start_x + j * 0.1
            centers[current_idx] = [x, y]
            current_idx += 1
            
    # 2. Optimization Loop (Force-Directed Layout)
    # We simulate the circles expanding and repelling each other
    max_iter = 2000
    temperature = 1.0
    
    for step in range(max_iter):
        # Calculate forces
        forces = np.zeros((n, 2))
        
        # Increase target radius slightly (expansion)
        # Target radius increases logarithmically to slow down over time
        expansion_rate = 0.00005 * (1 / (1 + step * 0.001))
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # Avoid division by zero
                if dist < 1e-9:
                    dist = 1e-9
                    dx, dy = 1.0, 1.0 # Arbitrary direction if too close
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist:
                    # Overlap: Repel strongly
                    # Force proportional to overlap amount
                    overlap = min_dist - dist
                    repulsion = overlap * 10.0
                    
                    # Direction vector
                    fx = (dx / dist) * repulsion
                    fy = (dy / dist) * repulsion
                    
                    forces[i] -= [fx, fy]
                    forces[j] += [fx, fy]
        
        # Boundary forces
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (-(x - r)) * 10.0
            # Right wall
            if x + r > 1:
                forces[i, 0] -= ((x + r) - 1) * 10.0
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (-(y - r)) * 10.0
            # Top wall
            if y + r > 1:
                forces[i, 1] -= ((y + r) - 1) * 10.0
                
            # Expansion force: try to grow radius
            # If not touching anything, grow. If touching, growth is resisted by repulsion.
            # We can simulate growth by a small outward force from the center of mass?
            # Or simpler: just increase radius directly if it fits?
            # But we need to find the max. 
            # Better: apply a force pushing centers apart to "create space" for growth?
            # Actually, just increasing radius in the loop and relying on repulsion is standard.
        
        # Update centers
        step_size = 0.05 * temperature
        centers += forces * step_size
        
        # Clip centers to stay within square (prevent explosion)
        # Centers must be at least r away from boundaries? 
        # No, forces handle that. But for safety:
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])

        # Increase radii uniformly
        # Check if we can expand
        can_expand = True
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] + 1e-7:
                    can_expand = False
                    break
            if not can_expand: break
            
            # Check boundaries
            if centers[i, 0] - radii[i] < 1e-7 or centers[i, 0] + radii[i] > 1 - 1e-7:
                can_expand = False
            if centers[i, 1] - radii[i] < 1e-7 or centers[i, 1] + radii[i] > 1 - 1e-7:
                can_expand = False
            
        if can_expand:
            radii += expansion_rate
            
        # Cool down
        temperature *= 0.999

    # 3. Refine: Local Optimization for each circle
    # Try to push circles to boundaries to free up space in the middle
    for _ in range(500):
        for i in range(n):
            # Try to move circle i in random directions or towards "empty" space?
            # Simple gradient ascent on "slack"
            pass

    # 4. Final validation and correction
    # Ensure strict validity
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        # Clamp to boundary
        if x - r < 0: x = r
        if x + r > 1: x = 1 - r
        if y - r < 0: y = r
        if y + r > 1: y = 1 - r
        centers[i] = [x, y]

    # Resolve any remaining minor overlaps by shrinking radii slightly if needed
    # Iterative shrink
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-12:
                    overlap = (radii[i] + radii[j]) - dist
                    # Reduce radii to resolve overlap
                    # Distribute reduction
                    reduction = overlap / 2
                    radii[i] -= reduction
                    radii[j] -= reduction
                    changed = True
        
        # Boundary check
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x - r < 0 or x + r > 1 or y - r < 0 or y + r > 1:
                # Adjust radius to fit
                max_r_x = min(x, 1 - x)
                max_r_y = min(y, 1 - y)
                new_r = min(r, max_r_x, max_r_y)
                if new_r < r:
                    radii[i] = new_r
                    changed = True
        if not changed:
            break
            
    # Ensure non-negative radii
    radii = np.maximum(radii, 0)

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

if __name__ == "__main__":
    import numpy as np
    # Re-define validate_packing here to test locally if needed, 
    # but the prompt says it's read-only.
    # We just need to output the code for run_packing.
    pass
