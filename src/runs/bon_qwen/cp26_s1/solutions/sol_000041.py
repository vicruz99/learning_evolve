# sol_000041 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 634a18b7) state=75f7ce1d sum of radii=2.336618 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def generate_hexagonal_initial_positions(n):
    """
    Generates initial positions for n circles in a hexagonal pattern
    to serve as a good starting point for optimization.
    """
    # Estimate radius based on packing density
    # Area of square = 1. Density ~ 0.9. 
    # n * pi * r^2 ~ 0.9 => r ~ sqrt(0.9 / (n * pi))
    # For n=26, r ~ 0.105
    # However, boundaries reduce this. Let's start with r=0.08 to be safe.
    r_est = 0.08
    
    # We want to fit n circles. 
    # Let's try to fit them in a hexagonal grid.
    # Number of columns and rows.
    # Approximation: area per circle ~ r^2 * sqrt(3)
    # Width ~ sqrt(n) * r
    
    # Let's just place them in a grid and perturb, or generate hex coords
    centers = []
    
    # Try to fit in rows
    # Hexagonal packing: rows shifted by r, vertical dist r*sqrt(3)
    # Let's try to determine number of rows and cols
    # sqrt(26) approx 5.1. So maybe 5x5 or 6x5.
    
    # Let's create a grid of potential points and pick the best n
    # Or just place them.
    
    # Let's try a simple grid first, then perturb
    # 5x5 grid has 25 spots. We need 26.
    # Let's try 6 columns, 5 rows?
    # Width 1. 6 circles -> spacing 1/6 approx 0.16. Radius ~0.08.
    # Height 1. 5 rows -> spacing 1/5 = 0.2.
    
    # Let's generate points for a hexagonal lattice covering the square
    # and select n closest to center or just first n valid ones.
    
    # Parameters for hex lattice
    # Spacing dx = 2*r_est, dy = sqrt(3)*r_est
    # But we don't know optimal r_est exactly.
    # Let's assume r=0.09
    r_temp = 0.09
    dx = 2 * r_temp
    dy = math.sqrt(3) * r_temp
    
    points = []
    y = r_temp
    row_idx = 0
    while y < 1 - r_temp:
        x = r_temp
        if row_idx % 2 == 1:
            x += dx / 2 # Shift odd rows
        while x < 1 - r_temp:
            points.append((x, y))
            x += dx
        y += dy
        row_idx += 1
    
    # If we don't have enough points, relax r_temp
    while len(points) < n:
        r_temp *= 0.9
        dx = 2 * r_temp
        dy = math.sqrt(3) * r_temp
        points = []
        y = r_temp
        row_idx = 0
        while y < 1 - r_temp:
            x = r_temp
            if row_idx % 2 == 1:
                x += dx / 2
            while x < 1 - r_temp:
                points.append((x, y))
                x += dx
            y += dy
            row_idx += 1
            
    # Select first n points
    selected_points = points[:n]
    
    # If still not enough (unlikely), fill with random
    if len(selected_points) < n:
        while len(selected_points) < n:
            selected_points.append((np.random.rand(), np.random.rand()))
            
    return np.array(selected_points)

def validate_and_fix(centers, radii):
    """
    Helper to ensure radii are valid for current centers
    and clip if necessary.
    """
    n = centers.shape[0]
    # Clip radii to fit in square
    for i in range(n):
        x, y = centers[i]
        max_r_bound = min(x, 1-x, y, 1-y)
        if max_r_bound < 0:
            max_r_bound = 0
        radii[i] = min(radii[i], max_r_bound)
        
    # Clip radii to avoid overlap (simple fix: if overlap, reduce radius)
    # This is expensive O(N^2), but N=26 is small.
    # We do this iteratively or just return. 
    # Better to let optimizer handle it, but this helps stability.
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < radii[i] + radii[j] - 1e-9:
                    # Overlap detected. Reduce both radii to touch
                    # Keep ratio or just split?
                    # Simple: scale down sum of radii to match distance
                    current_sum = radii[i] + radii[j]
                    if current_sum > 0:
                        scale = dist / current_sum
                        radii[i] *= scale
                        radii[j] *= scale
                        changed = True
    return radii

def run_packing():
    np.random.seed(42) # For reproducibility
    n = 26
    
    # 1. Initial placement
    centers = generate_hexagonal_initial_positions(n)
    # Start with small radii
    radii = np.full(n, 0.05)
    
    # 2. Iterative expansion
    # We try to expand radii and move centers to accommodate
    for step in range(100):
        # Calculate max possible radius for each circle based on current centers
        # This is an LP, but we can approximate greedily
        # r_i <= dist(i,j) - r_j
        # We can just set r_i to min distance to others / 2 + ...?
        # Better: calculate clearance
        clearances = np.full(n, 1.0)
        
        # Boundary clearance
        for i in range(n):
            x, y = centers[i]
            clearances[i] = min(x, 1-x, y, 1-y)
            
        # Neighbor clearance
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                # If we increase radii equally, we can increase by (dist - (r_i + r_j))/2
                # But here we just check if current radii are too big
                # Or compute max allowed r_i given r_j
                # Let's just compute available space
                if dist < 1e-6:
                    dist = 1e-6
                
                # Max r_i is dist - r_j
                max_r_i = dist - radii[j]
                if max_r_i < clearances[i]:
                    clearances[i] = max_r_i
        
        # Update radii to max possible (clamped to current + growth)
        # To avoid shrinking too much, we can only increase?
        # But we might need to shrink if centers moved bad.
        # Let's just set radii to clearances.
        # But this might cause oscillation.
        # Let's take average of current and clearance?
        # Actually, if we set radii = clearances, they will touch.
        # Then we need to move centers apart.
        
        # Let's perform a force-directed step
        force = np.zeros_like(centers)
        k = 0.1 # Spring constant
        repulsion_strength = 0.01
        
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    dist = 1e-6
                
                # Ideal distance is radii[i] + radii[j]
                # But we want to increase radii, so we want dist to be large.
                # Repulsive force if dist < sum_radii + margin
                sum_r = radii[i] + radii[j]
                target_dist = sum_r + 0.01 # Push apart slightly
                
                if dist < target_dist:
                    # Force proportional to penetration
                    rep_force = (target_dist - dist) * repulsion_strength
                    direction = diff / dist
                    force[i] += direction * rep_force
                    force[j] -= direction * rep_force
            
            # Boundary repulsion
            x, y = centers[i]
            r = radii[i]
            # Push away from boundaries
            # Left
            if x - r < 0:
                force[i, 0] += (r - x) * k
            # Right
            if x + r > 1:
                force[i, 0] -= (x + r - 1) * k
            # Bottom
            if y - r < 0:
                force[i, 1] += (r - y) * k
            # Top
            if y + r > 1:
                force[i, 1] -= (y + r - 1) * k
        
        # Update centers
        step_size = 0.05 / (1 + step * 0.01) # Decaying step size
        centers += force * step_size
        
        # Clip centers to valid range [0, 1]
        centers = np.clip(centers, 1e-5, 1 - 1e-5)
        
        # Update radii to max possible given new centers
        # We want to maximize sum of radii.
        # If centers moved apart, we can increase radii.
        # Let's compute max valid radius for each
        new_radii = np.full(n, 1.0)
        
        # Boundary constraints
        for i in range(n):
            x, y = centers[i]
            new_radii[i] = min(x, 1-x, y, 1-y)
        
        # Overlap constraints
        # r_i + r_j <= dist_ij
        # This is a system. r_i <= dist_ij - r_j.
        # We can iterate to solve this or just do one pass.
        # One pass: r_i = min(r_i, dist_ij - r_j) for all j.
        # Order matters. Let's do a few passes.
        for _ in range(10):
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    dist = np.linalg.norm(centers[i] - centers[j])
                    if dist < 1e-9: continue
                    # r_i <= dist - r_j
                    # But r_j is changing. 
                    # We can enforce r_i + r_j <= dist
                    # If r_i + r_j > dist, reduce the larger one?
                    # Or split reduction.
                    # Simple: r_i = min(r_i, dist - new_radii[j])
                    # This is biased.
                    pass 
            # Better: just compute min dist to neighbors
            for i in range(n):
                min_d = 1.0
                for j in range(n):
                    if i == j: continue
                    d = np.linalg.norm(centers[i] - centers[j])
                    # If we assume neighbors have radius r_j, max r_i is d - r_j
                    # But we don't know optimal r_j yet.
                    # However, if we set r_i = d/2, we satisfy r_i+r_j <= d if r_j=d/2.
                    # This is a safe lower bound.
                    if d < min_d:
                        min_d = d
                new_radii[i] = min(new_radii[i], min_d / 2)
        
        # We can actually increase radii if there is slack.
        # But min_d/2 is conservative (assumes equal neighbors).
        # If neighbors are small, we can be bigger.
        # Let's try to be optimistic: r_i = min(dist - r_j) ?
        # But r_j are the *new* radii?
        # Let's use the previous radii as estimate for neighbors?
        # r_i = min(dist - radii[j])
        # This might increase r_i if radii[j] is small.
        
        # Let's refine radii calculation
        # We want to maximize sum. 
        # If we have space, increase r.
        for i in range(n):
            max_r = new_radii[i] # From boundary
            for j in range(n):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                # Constraint: r_i + radii[j] <= dist  (using old radii for j?)
                # Or r_i + new_radii[j] <= dist?
                # If we use new_radii[j] (which is conservative), we are safe.
                # But new_radii[j] might be smaller than optimal.
                # Let's use current radii array which holds previous valid radii.
                # But we updated centers, so radii might be invalid.
                # Let's recompute valid radii from scratch using a fixed point iteration.
                pass
            
            # Let's just use the boundary limit for now and let optimizer fix overlaps
            # But we need valid packing for validate_packing.
            # So we must ensure non-overlap.
            
            # Let's enforce: r_i = min(r_i, dist_ij - r_j)
            # We can do this in a loop until convergence.
            changed = True
            temp_radii = new_radii.copy()
            while changed:
                changed = False
                for i in range(n):
                    current_r = temp_radii[i]
                    for j in range(n):
                        if i == j: continue
                        dist = np.linalg.norm(centers[i] - centers[j])
                        # Required: r_i + r_j <= dist
                        # r_i <= dist - r_j
                        limit = dist - temp_radii[j]
                        if current_r > limit + 1e-9:
                            temp_radii[i] = limit
                            changed = True
                            if temp_radii[i] < 0: temp_radii[i] = 0
            radii = temp_radii

    # 3. Final Optimization with Scipy
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # But we have 26 circles. 78 vars.
    # Constraints are many.
    # We can try to optimize just centers assuming radii are computed optimally?
    # Or optimize both.
    # Given time, let's try to optimize centers only, with radii recomputed?
    # No, objective depends on radii.
    
    # Let's use SLSQP on a reduced set or just try it.
    # To speed up, we can assume radii are equal? No.
    
    # Let's try a local optimization of centers.
    # Objective: sum of radii(centers)
    # But radii(centers) is non-smooth (min of distances).
    # However, we can smooth it or just use the discrete nature.
    
    # Actually, the force-directed step already did a lot.
    # Let's just clean up the radii to be valid and return.
    # The force directed step should have increased sum of radii.
    
    # One last check: ensure valid
    radii = validate_and_fix(centers, radii)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# To make it runnable as requested
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
