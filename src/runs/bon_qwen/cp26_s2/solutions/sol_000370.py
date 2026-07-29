# sol_000370 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b75b923f) state=e2452a32 sum of radii=2.166663 correctness=1.0
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
    centers = np.zeros((n, 2))
    
    # 1. Initialization: Hexagonal-like grid
    # We try to fit 26 circles. 5 rows is a good height.
    # Distribution: 6, 5, 5, 5, 5 = 26 circles.
    # We place them in the square [0,1]x[0,1].
    
    # Approximate positions
    # We use a spacing slightly larger than expected optimal to allow relaxation
    # Expected radius ~ 0.1. Diameter ~ 0.2.
    
    current_r = 0.1
    row_counts = [6, 5, 5, 5, 5]
    
    idx = 0
    # Vertical spacing for hexagonal packing is r * sqrt(3)
    # Total height for 5 rows: 2*r (margins) + 4 * r*sqrt(3)
    # We scale this to fit in [0,1]
    # Let's just distribute y uniformly first and let optimization fix it
    
    for r_idx, count in enumerate(row_counts):
        # y coordinate for this row
        # Spread rows evenly
        y = (r_idx + 0.5) / 5.0 
        
        # x coordinates
        # Stagger rows: even rows aligned, odd rows shifted?
        # Or just center them.
        
        # Width for 'count' circles with radius r: 2*r*count
        # We want to fit in [0,1]. 
        # Let's place them centered.
        
        # Horizontal spacing
        if count > 1:
            spacing = (1 - 2*current_r) / (count - 1)
            # If spacing is too small, circles will overlap initially, which is fine.
            # But let's ensure they fit roughly.
            # If count is 6, 2*0.1*6 = 1.2 > 1. So we squeeze.
            # We will rely on optimizer to spread them.
            # Let's just place them evenly in [current_r, 1-current_r]
            start_x = current_r
            end_x = 1 - current_r
            xs = np.linspace(start_x, end_x, count)
        else:
            xs = [0.5]
            
        # Shift odd rows?
        # If we shift, we might hit walls.
        # Let's try a slight shift for hexagonal feel
        if r_idx % 2 == 1:
            shift = spacing / 2.0
            xs = xs + shift
            # Clip to valid range just in case
            xs = np.clip(xs, current_r, 1 - current_r)

        for x in xs:
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
    
    # Initialize radii
    radii = np.full(n, current_r)
    
    # 2. Optimization (Repulsion Simulation)
    # We will try to expand radii and resolve conflicts
    
    # Parameters
    iterations = 2000
    learning_rate = 0.01
    repulsion_strength = 1.0
    
    # We will run a loop to increase radius and optimize positions
    # A simple strategy: fix radii, optimize positions to minimize overlap/penalty
    # Then increase radii slightly.
    
    # However, simpler: Just run a force-based simulation with variable radii?
    # Let's stick to: Find max r for equal circles first? 
    # But unequal might be better.
    # Let's assume equal radii for simplicity of robust optimization, 
    # as unequal radii optimization is complex and might not gain much for N=26.
    # Actually, for sum of radii, equal is often near optimal.
    
    # Let's try to maximize equal radius r.
    # We use a binary search on r? No, continuous optimization is better.
    
    # Let's use a simple gradient-free optimization:
    # At each step, calculate "pressure" on each circle.
    # Pressure comes from neighbors and walls.
    # Move centers to relieve pressure.
    # Also expand radii if no pressure.
    
    # Better: Use a "simulated annealing" or "hill climbing" on the configuration space?
    # Too complex.
    
    # Let's use a specialized solver logic:
    # 1. Start with r=0.05.
    # 2. Place circles.
    # 3. While True:
    #    Try to increase r by delta.
    #    Run local optimization to fix overlaps.
    #    If fails to fix, rollback or stop.
    
    # Let's implement a robust local optimization.
    
    # Reset centers to a valid state with small r
    radii[:] = 0.05
    # Re-init centers nicely
    idx = 0
    for r_idx, count in enumerate(row_counts):
        y = (r_idx + 0.5) / 5.0
        if count > 1:
            xs = np.linspace(0.1, 0.9, count) # Loose spacing
        else:
            xs = [0.5]
        if r_idx % 2 == 1:
            shift = 0.05
            xs = xs + shift
        for x in xs:
            if idx < n:
                centers[idx] = [np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)]
                idx += 1
                
    # Optimization Loop
    # We will iterate to expand radii
    
    max_r = 0.12 # Upper bound estimate
    current_r = 0.05
    
    # We can optimize for equal radii first
    # Then maybe allow small variations? 
    # Let's stick to equal radii optimization to ensure validity and high sum.
    # Sum = 26 * r.
    
    # Function to check if a configuration is valid for a given r
    # and to resolve overlaps by moving centers.
    
    def resolve_overlaps(centers, r, steps=100):
        """
        Try to move centers to resolve overlaps for given radius r.
        Returns (success, new_centers)
        """
        n = centers.shape[0]
        # Copy centers
        c = centers.copy()
        
        for _ in range(steps):
            moved = False
            for i in range(n):
                # Check boundaries
                dx = 0
                dy = 0
                if c[i, 0] - r < 0: dx += r - (c[i, 0] - r)
                if c[i, 0] + r > 1: dx -= (c[i, 0] + r - 1)
                if c[i, 1] - r < 0: dy += r - (c[i, 1] - r)
                if c[i, 1] + r > 1: dy -= (c[i, 1] + r - 1)
                
                if dx != 0 or dy != 0:
                    c[i, 0] += dx
                    c[i, 1] += dy
                    moved = True
                
                # Check pairwise overlaps
                for j in range(i + 1, n):
                    dist_vec = c[i] - c[j]
                    dist = np.linalg.norm(dist_vec)
                    min_dist = 2 * r
                    if dist < min_dist and dist > 1e-9:
                        # Overlap! Push apart
                        overlap = min_dist - dist
                        push_vec = (dist_vec / dist) * (overlap / 2.0)
                        c[i] += push_vec
                        c[j] -= push_vec
                        moved = True
                    elif dist < 1e-9:
                        # Same point, push randomly
                        angle = np.random.uniform(0, 2 * np.pi)
                        offset = 0.01
                        c[i] += [offset * np.cos(angle), offset * np.sin(angle)]
                        c[j] -= [offset * np.cos(angle), offset * np.sin(angle)]
                        moved = True
            
            if not moved:
                break
        
        # Verify validity
        valid = True
        for i in range(n):
            if c[i, 0] - r < -1e-9 or c[i, 0] + r > 1 + 1e-9 or \
               c[i, 1] - r < -1e-9 or c[i, 1] + r > 1 + 1e-9:
                valid = False
                break
            for j in range(i + 1, n):
                dist = np.linalg.norm(c[i] - c[j])
                if dist < 2 * r - 1e-9:
                    valid = False
                    break
            if not valid: break
            
        return valid, c

    # Search for maximum r
    # Binary search or iterative increase
    low = 0.05
    high = 0.12
    
    # First, find a valid r
    # Try to increase r step by step
    r_try = 0.05
    # Initialize with a good config
    # Let's run resolve_overlaps a few times to settle
    
    # Better: Use a loop to find max r
    # We start small and grow.
    
    # Re-init centers for the growth process
    # Use the hex grid
    idx = 0
    for r_idx, count in enumerate(row_counts):
        y = (r_idx + 0.5) / 5.0
        if count > 1:
            xs = np.linspace(0.1, 0.9, count)
        else:
            xs = [0.5]
        if r_idx % 2 == 1:
            xs = xs + 0.05
        for x in xs:
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
                
    radii[:] = 0.01 # Start very small
    
    # Grow radii
    step = 0.0005
    # We want to reach ~0.101
    # Max steps approx (0.101 - 0.01) / 0.0005 = 182 steps.
    # Each step runs resolution.
    
    # To make it faster, we can use larger steps initially.
    
    for _ in range(3000): # Max iterations
        # Try to increase radius
        proposed_r = radii[0] + step
        
        # Check if we can resolve overlaps with proposed_r
        # We need to run resolve_overlaps
        # But resolve_overlaps is expensive (O(N^2 * steps)).
        # Let's optimize: only run if necessary?
        # Actually, let's just run it. 26^2 is small.
        
        success, new_centers = resolve_overlaps(centers, proposed_r, steps=50)
        
        if success:
            radii[:] = proposed_r
            centers = new_centers
            # Occasionally increase step size to speed up
            if step < 0.002:
                step *= 1.01
        else:
            # If failed, decrease step size and try again or stop?
            # Decrease step
            step /= 1.5
            if step < 1e-6:
                break
            # Also, maybe the current centers are not optimal for current radius.
            # Try to optimize current radius first?
            # Let's do a local optimization pass for current radius
            success, centers = resolve_overlaps(centers, radii[0], steps=100)
            if not success:
                # Should not happen if we came from a valid state
                break
    
    # Final polish: try to squeeze a bit more
    # Current r is valid. Try to go higher with smaller steps
    step = 1e-4
    for _ in range(100):
        proposed_r = radii[0] + step
        success, new_centers = resolve_overlaps(centers, proposed_r, steps=100)
        if success:
            radii[:] = proposed_r
            centers = new_centers
        else:
            step /= 2
            if step < 1e-7:
                break

    # Calculate sum
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
