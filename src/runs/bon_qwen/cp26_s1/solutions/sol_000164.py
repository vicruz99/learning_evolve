# sol_000164 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1d84d4eb) state=42faed8c sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    np.random.seed(42) # For reproducibility
    n = 26
    
    # 1. Initialize centers in a dense hexagonal-like grid
    # We try to fit 26 circles. A 5x5 grid is 25, so we add one or adjust.
    # Let's try a layout that roughly fits 26. 
    # 5 rows, alternating 5 and 6 circles might be tight but a good start.
    # Or just a dense random packing.
    # Let's use a staggered grid initialization.
    
    centers = np.zeros((n, 2))
    idx = 0
    
    # Attempt to place circles in a pattern
    # 5 rows
    rows = 5
    cols_per_row = [6, 5, 6, 5, 4] # Sum = 26. 6 is tight, might need to shrink.
    # Actually 6 circles of radius 0.1 needs width 1.2, which is too big.
    # Let's use a more balanced grid first, then optimize.
    # 5x5 grid + 1 extra in a gap?
    # Let's just use a 6x5 grid (30 spots) and remove 4, or just place 26 in a rectangle.
    
    # Let's try a 5x6 grid arrangement (5 rows, 6 cols) but we only need 26.
    # We can just fill the first 26 spots of a 6x5 grid.
    # 6 columns, 5 rows.
    # Spacing: dx = 1/6, dy = 1/5 ? No, we need to leave margin.
    # Let's just use a random dense start.
    
    # Better initialization: Hexagonal packing pattern
    # We can estimate optimal r ~ 0.105.
    # Lattice spacing ~ 2r ~ 0.21.
    # Row spacing ~ r*sqrt(3) ~ 0.182.
    
    r_est = 0.105
    centers = []
    
    # Try to pack in a hexagonal lattice
    y = r_est
    row = 0
    while len(centers) < n and y + r_est <= 1:
        x = r_est
        offset = r_est if row % 2 == 1 else 0
        x = r_est + offset
        while x + r_est <= 1 and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_est
        y += r_est * np.sqrt(3)
        row += 1
        
    # If we didn't fit enough, fill remaining randomly or adjust
    if len(centers) < n:
        # Fill remaining with random positions in safe zone
        for i in range(len(centers), n):
            cx = np.random.uniform(0.1, 0.9)
            cy = np.random.uniform(0.1, 0.9)
            centers.append([cx, cy])
            
    centers = np.array(centers[:n])
    
    # Initialize radii
    radii = np.full(n, r_est)
    
    # 2. Optimization
    # We will maximize sum(radii) by adjusting centers and radii.
    # Since direct optimization is hard, we use an iterative expansion.
    
    # To make it robust, we can use a simple gradient ascent on sum(radii)
    # with repulsion forces.
    
    # However, a simpler approach for coding:
    # Use scipy to minimize negative sum of radii with constraints?
    # Constraints are complex.
    # Let's stick to the iterative geometric push.
    
    # Iterative Refinement
    for step in range(2000):
        # Calculate current valid radii for fixed centers
        # This is a linear programming problem, but we can approximate.
        # Actually, for a fixed set of centers, the max radius for circle i 
        # is min(dist to boundary, min_j (dist_ij - r_j)).
        # This is coupled. 
        # But if we assume radii are roughly equal, we can find max r.
        # Let's just try to increase radii slightly and fix overlaps.
        
        # Try to increase radii
        radii *= 1.001
        
        # Check constraints and resolve conflicts
        # If a circle is too big (overlap or out of bounds), move it or shrink it.
        
        # To avoid shrinking, let's move centers to resolve overlaps.
        # Force-directed layout.
        
        forces = np.zeros_like(centers)
        
        # Boundary forces (push away from edges)
        # If circle is outside or touching boundary with r, push in.
        # Ideally centers should be in [r, 1-r].
        # If center < r, push right. If center > 1-r, push left.
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left
            if x < r:
                forces[i, 0] += (r - x) * 10.0
            # Right
            elif x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * 10.0
            # Bottom
            if y < r:
                forces[i, 1] += (r - y) * 10.0
            # Top
            elif y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * 10.0
                
        # Repulsion forces for overlaps
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                req_dist = radii[i] + radii[j]
                if dist < req_dist:
                    # Overlap
                    overlap = req_dist - dist
                    # Force proportional to overlap
                    # Direction from j to i
                    if dist > 1e-9:
                        dir_vec = (centers[i] - centers[j]) / dist
                        force_mag = overlap * 5.0 # Spring constant
                        forces[i] += dir_vec * force_mag
                        forces[j] -= dir_vec * force_mag
                    else:
                        # Same center, push randomly
                        forces[i] += np.random.randn(2) * 0.1
                        forces[j] -= np.random.randn(2) * 0.1
        
        # Apply forces
        # Learning rate
        lr = 0.5 / (1 + step / 100) # Decay learning rate
        centers += forces * lr
        
        # Clamp centers to [0, 1]
        centers = np.clip(centers, 0, 1)
        
        # After moving, we might need to adjust radii if they are too large for new positions
        # But we just increased radii at the start of loop.
        # If forces resolved overlap, radii might be valid.
        # However, if a circle hits boundary, it might need to shrink.
        # But our force pushes it inside, so it should be valid eventually.
        
        # To ensure strict validity at the end, we can compute max valid radii.
        
    # Final cleanup: Compute maximum valid radii for the final centers
    # This ensures we return a strictly valid packing.
    # We can solve this by iterative relaxation.
    
    # Initialize radii to small value
    final_radii = np.full(n, 0.01)
    
    # Iterate to find max radii
    for _ in range(100):
        new_radii = np.copy(final_radii)
        for i in range(n):
            x, y = centers[i]
            # Boundary constraints
            r_bound = min(x, 1 - x, y, 1 - y)
            r_bound = max(r_bound, 0)
            
            # Neighbor constraints
            r_neighbor = r_bound
            for j in range(n):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                # r_i + r_j <= dist  => r_i <= dist - r_j
                max_r = dist - final_radii[j]
                if max_r < r_neighbor:
                    r_neighbor = max_r
            
            new_radii[i] = min(r_bound, max(0, r_neighbor))
        final_radii = new_radii

    # Verify and adjust if necessary
    # Sometimes the iterative relaxation might be slightly off due to order of updates
    # Let's do a safety check and shrink if needed
    
    # Check overlaps and shrink
    for _ in range(50):
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < final_radii[i] + final_radii[j] - 1e-9:
                    valid = False
                    # Shrink both
                    excess = (final_radii[i] + final_radii[j]) - dist
                    factor = excess / (final_radii[i] + final_radii[j]) if (final_radii[i] + final_radii[j]) > 0 else 0
                    final_radii[i] *= (1 - factor)
                    final_radii[j] *= (1 - factor)
        
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            r = final_radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                valid = False
                # Shrink
                margin = max(x - r, r - x, y - r, r - y, r - (1-y), (1-y) - r) # logic error here
                # Actually just clamp radius
                max_r = min(x, 1-x, y, 1-y)
                if final_radii[i] > max_r:
                    final_radii[i] = max_r

        if valid:
            break

    sum_radii = np.sum(final_radii)
    
    return centers, final_radii, sum_radii

# Run the packing to get the result
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic validation
    try:
        from numpy import sqrt
        # Check overlaps manually
        for i in range(len(c)):
            for j in range(i+1, len(c)):
                dist = sqrt(np.sum((c[i]-c[j])**2))
                if dist < r[i] + r[j] - 1e-9:
                    print(f"Overlap {i},{j}")
        for i in range(len(c)):
            if r[i] < 0 or c[i][0]-r[i]<-1e-9 or c[i][0]+r[i]>1+1e-9 or c[i][1]-r[i]<-1e-9 or c[i][1]+r[i]>1+1e-9:
                print(f"Invalid {i}")
    except Exception as e:
        print(e)
