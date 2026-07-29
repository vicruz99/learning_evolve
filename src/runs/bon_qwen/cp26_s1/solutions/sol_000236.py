# sol_000236 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dc099519) state=5317863c sum of radii=0.026000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import random

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # We will try multiple restarts with different initial configurations
    # and a local optimization to maximize sum of radii.
    
    def get_sum_radii(centers, radii):
        # Check validity
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            if x < r - 1e-7 or x > 1 - r + 1e-7 or y < r - 1e-7 or y > 1 - r + 1e-7:
                return -1.0 # Invalid
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-7:
                    return -1.0 # Invalid
        return np.sum(radii)

    # Optimization function
    def optimize(centers, radii, steps=500):
        # We want to maximize sum(radii).
        # We can use a repulsive force model.
        # If circles overlap, push them apart.
        # If not, allow them to grow.
        
        # To make it easier, we can fix radii to be equal during optimization?
        # No, let's allow variation but start with equal.
        
        # Current radii
        r = radii.copy()
        c = centers.copy()
        
        # We try to expand radii by a small amount and resolve conflicts
        expansion_rate = 0.001
        
        for _ in range(steps):
            # Try to increase radii
            # If a circle is not constrained by boundary or neighbors, expand it.
            # But we need a global approach.
            
            # Let's use a force-based approach.
            # Forces:
            # 1. Repulsion between overlapping circles.
            # 2. Repulsion from boundaries if r is too large.
            # 3. "Gravity" to center? No.
            
            # Instead, let's fix the radius r and find positions.
            # Then binary search for max r.
            # But circles might be different sizes.
            
            # Let's assume equal radii for simplicity first, as it's likely optimal.
            # Maximize r such that 26 circles of radius r fit.
            
            # For a fixed r, check if valid.
            # We can use a repulsive force to arrange centers.
            
            # Current r
            curr_r = np.mean(r)
            
            # Apply forces
            forces = np.zeros_like(c)
            
            for i in range(n):
                for j in range(i + 1, n):
                    dx = c[i][0] - c[j][0]
                    dy = c[i][1] - c[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    min_dist = 2 * curr_r # Target distance
                    if dist < min_dist and dist > 1e-7:
                        repulsion = (min_dist - dist) * 0.1 # Strength
                        fx = repulsion * dx / dist
                        fy = repulsion * dy / dist
                        forces[i][0] += fx
                        forces[i][1] += fy
                        forces[j][0] -= fx
                        forces[j][1] -= fy
                
                # Boundary forces
                x, y = c[i]
                if x - curr_r < 0:
                    forces[i][0] += curr_r * 0.5 # Push in
                elif x + curr_r > 1:
                    forces[i][0] -= curr_r * 0.5
                if y - curr_r < 0:
                    forces[i][1] += curr_r * 0.5
                elif y + curr_r > 1:
                    forces[i][1] -= curr_r * 0.5
            
            c += forces * 0.1 # Move
            
            # Clip to valid range (loosely)
            c = np.clip(c, 0.0, 1.0)
            
            # Try to increase r slightly if no overlaps?
            # Check overlaps
            has_overlap = False
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.sqrt(np.sum((c[i] - c[j])**2))
                    if dist < 2 * curr_r - 1e-5:
                        has_overlap = True
                        break
                if has_overlap: break
            
            # Check boundaries
            out = False
            for i in range(n):
                if c[i][0] < curr_r - 1e-5 or c[i][0] > 1 - curr_r + 1e-5 or \
                   c[i][1] < curr_r - 1e-5 or c[i][1] > 1 - curr_r + 1e-5:
                    out = True
                    break
            
            if not has_overlap and not out:
                curr_r += 0.0001 # Expand
            else:
                curr_r -= 0.00005 # Shrink if needed
            
            r[:] = curr_r
            
        return c, r

    # Try multiple random inits
    best_valid_sum = 0
    best_valid_c = None
    best_valid_r = None
    
    for _ in range(10):
        # Init centers
        c = np.random.rand(n, 2)
        r = np.ones(n) * 0.05
        
        c, r = optimize(c, r, steps=1000)
        
        # Check final validity and sum
        s = get_sum_radii(c, r)
        if s > 0 and s > best_valid_sum:
            best_valid_sum = s
            best_valid_c = c.copy()
            best_valid_r = r.copy()
            
        # If we reached a high sum, keep it.
        
    # If optimization failed to find a valid packing, fallback to 5x5 grid + 1
    if best_valid_sum < 2.5:
        # Fallback: 5x5 grid of 0.099 radius (to be safe)
        xs = [0.1, 0.3, 0.5, 0.7, 0.9]
        ys = [0.1, 0.3, 0.5, 0.7, 0.9]
        c = []
        for y in ys:
            for x in xs:
                c.append([x, y])
        # 26th circle?
        # Try to fit in a gap.
        # Gap at (0.2, 0.2) distance to (0.1,0.1) is 0.141.
        # 2*0.099 = 0.198. Overlap.
        # So 5x5 grid of 0.1 is max for 25.
        # For 26, we need smaller.
        # Let's just return the optimized result if valid.
        
        # Fallback to random valid packing
        c = np.random.rand(n, 2)
        r = np.ones(n) * 0.01
        best_valid_c = c
        best_valid_r = r

    # Final validation
    if get_sum_radii(best_valid_c, best_valid_r) > 0:
        return best_valid_c, best_valid_r, best_valid_sum
    else:
        # Return a safe small packing
        c = np.random.rand(n, 2)
        r = np.ones(n) * 0.001
        return c, r, np.sum(r)
