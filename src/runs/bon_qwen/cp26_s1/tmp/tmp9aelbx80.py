import numpy as np
import math
import random

def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def min_distance_to_boundary(p):
    return min(p[0], 1.0-p[0], p[1], 1.0-p[1])

def optimize_positions(centers, n_iter=2000, cooling=0.99):
    """
    Optimize positions of points to maximize the minimum distance between them
    and from the boundaries.
    """
    n = centers.shape[0]
    centers = centers.astype(float)
    
    # Initial step size
    step = 0.01
    temp = 1.0
    
    for _ in range(n_iter):
        moved = False
        # Calculate forces
        # We want to maximize min distance.
        # Heuristic: Repulsive forces proportional to 1/d^2 or similar.
        # But to specifically target the min distance, we can push the closest pairs apart more.
        
        # Calculate all pairwise distances
        dists = np.zeros(n)
        for i in range(n):
            d_boundary = min_distance_to_boundary(centers[i])
            d_min = d_boundary
            for j in range(n):
                if i == j: continue
                d = distance(centers[i], centers[j])
                if d < d_min:
                    d_min = d
            dists[i] = d_min
            
        # Find the bottleneck (smallest distance)
        min_d = np.min(dists)
        
        # If min_d is very small, apply strong repulsion
        # If min_d is large, we are done?
        
        # Apply forces
        forces = np.zeros_like(centers)
        
        # Repulsion from other points
        for i in range(n):
            for j in range(i+1, n):
                p1 = centers[i]
                p2 = centers[j]
                d = distance(p1, p2)
                if d < 0.001: # Prevent division by zero
                    d = 0.001
                # Force magnitude
                # We want to push apart if d is small.
                # A force like 1/d^2 is standard for particle systems.
                # To focus on max-min, maybe use a threshold?
                # But simple repulsion works well to spread points.
                f_mag = 1.0 / (d**2) 
                
                # Direction
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                if d > 1e-9:
                    fx = (dx / d) * f_mag
                    fy = (dy / d) * f_mag
                else:
                    fx, fy = 0, 0
                
                forces[i][0] += fx
                forces[i][1] += fy
                forces[j][0] -= fx
                forces[j][1] -= fy
        
        # Repulsion from boundaries
        for i in range(n):
            x, y = centers[i]
            r_boundary = 0.01 # Virtual radius for boundary repulsion
            # Distance to left
            if x < r_boundary + 0.05:
                forces[i][0] += 1.0 / ((x + 0.001)**2) # Push right
            if x > 1.0 - r_boundary - 0.05:
                forces[i][0] -= 1.0 / ((1.0 - x + 0.001)**2) # Push left
            if y < r_boundary + 0.05:
                forces[i][1] += 1.0 / ((y + 0.001)**2) # Push up
            if y > 1.0 - r_boundary - 0.05:
                forces[i][1] -= 1.0 / ((1.0 - y + 0.001)**2) # Push down

        # Normalize forces to avoid explosion
        norms = np.linalg.norm(forces, axis=1, keepdims=True)
        norms[norms == 0] = 1
        forces = forces / norms * step # Use fixed step magnitude based on direction
        
        # Update positions
        for i in range(n):
            centers[i] += forces[i]
            
            # Clamp to [0, 1]
            centers[i][0] = np.clip(centers[i][0], 0.0, 1.0)
            centers[i][1] = np.clip(centers[i][1], 0.0, 1.0)
            
        # Decrease step size (cooling)
        step *= cooling
        if step < 1e-6:
            break
            
    return centers

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Initialization
    # Start with a hexagonal-like pattern or random
    # 26 points.
    # A 5x5 grid is 25 points. Add one in the center?
    # Or just random. Random is risky but with optimization it might work.
    # Let's try a structured init.
    
    centers = np.zeros((26, 2))
    idx = 0
    
    # Try to place in a hexagonal lattice pattern
    # Approximate radius 0.1. Spacing 0.2.
    # Rows
    y = 0.1
    row_idx = 0
    while idx < 26:
        # Determine x start
        if row_idx % 2 == 0:
            x_start = 0.1
            count = 5
        else:
            x_start = 0.2
            count = 5 # Try to fit 5, might need to adjust
            
        # Check width
        # If row_idx is odd, centers at 0.2, 0.4, 0.6, 0.8, 1.0? 
        # 1.0 is boundary. Radius 0.1 -> center at 0.9 max?
        # So offset row can fit fewer.
        # Let's just place 5 if possible, else 4.
        
        current_x = x_start
        placed_in_row = 0
        while placed_in_row < count and idx < 26:
            if current_x + 0.1 <= 1.0: # Ensure fits in square with r=0.1
                centers[idx][0] = current_x
                centers[idx][1] = y
                idx += 1
                placed_in_row += 1
                current_x += 0.2
            else:
                break
        y += 0.1732 # sqrt(3)/2 * 0.2 approx
        row_idx += 1
        
    # If we didn't fill 26, fill remaining randomly
    while idx < 26:
        centers[idx][0] = random.uniform(0.1, 0.9)
        centers[idx][1] = random.uniform(0.1, 0.9)
        idx += 1

    # 2. Optimize positions to maximize min-distance
    # Run multiple restarts to find good configuration
    best_centers = None
    best_min_d = -1.0
    
    # We will run the optimization on the initialized centers
    # And maybe a few random restarts
    
    # Strategy:
    # 1. Optimize the structured centers.
    # 2. Optimize a few random centers.
    # 3. Pick the one with highest min-distance.
    
    candidates = []
    candidates.append(centers)
    
    # Add random candidates
    for _ in range(5):
        rand_centers = np.random.rand(26, 2)
        # Scale to be inside
        rand_centers = 0.1 + 0.8 * rand_centers
        candidates.append(rand_centers)
        
    best_obj = -1.0
    
    for cand in candidates:
        opt_centers = optimize_positions(cand, n_iter=3000, cooling=0.995)
        
        # Calculate min distance for this configuration
        min_d = 1.0
        n = opt_centers.shape[0]
        for i in range(n):
            # Boundary
            d_bound = min_distance_to_boundary(opt_centers[i])
            if d_bound < min_d:
                min_d = d_bound
            # Pairwise
            for j in range(i+1, n):
                d = distance(opt_centers[i], opt_centers[j])
                if d < min_d:
                    min_d = d
        
        # We want to maximize min_d.
        # However, we can also calculate the potential sum of radii.
        # If min_d is the limiting factor, r = min_d / 2.
        # But we can do better with unequal radii.
        # Let's store the best configuration based on a heuristic.
        # Heuristic: sum of max possible radii.
        
        # Calculate max radii for fixed centers
        radii = np.zeros(n)
        for i in range(n):
            r = min_distance_to_boundary(opt_centers[i])
            for j in range(n):
                if i == j: continue
                d = distance(opt_centers[i], opt_centers[j])
                if d/2 < r:
                    r = d/2
            radii[i] = r
        
        sum_r = np.sum(radii)
        
        # We prefer higher sum_r.
        # But we need to make sure radii are valid (non-overlap).
        # The calculated radii are valid for the fixed centers.
        # But maybe we can improve centers further?
        
        if sum_r > best_obj:
            best_obj = sum_r
            best_centers = opt_centers.copy()
            best_radii = radii.copy()

    # 3. Refinement
    # Now we have a good set of centers and radii.
    # The radii are computed assuming fixed centers.
    # We can try to expand radii and move centers slightly.
    # But the force simulation already pushed centers apart.
    # The radii calculated are the maximal for those centers.
    # Is it possible to increase sum by moving centers?
    # The force simulation maximized min-distance, which is a good proxy.
    
    # Let's do a final check and adjustment.
    # We have best_centers and best_radii.
    # Check if valid.
    # If valid, return.
    
    # One small optimization: 
    # The radii are determined by the tightest constraint.
    # Some circles might be able to grow if we shrink others? 
    # But sum of radii is what we want.
    # Usually, in a "jammed" state from max-min distance, 
    # the circles are roughly equal size (radius = min_d/2).
    # If we have unequal radii, it's because of boundary effects or asymmetry.
    
    # Let's verify validity and compute sum.
    centers_final = best_centers
    radii_final = best_radii
    
    # Recalculate radii carefully to ensure no overlaps
    # Since we computed radii based on distances, they should be valid.
    # But due to floating point, maybe slightly reduce?
    # No, d/2 is safe.
    
    # Just to be super safe, ensure no overlaps with a small epsilon?
    # The validation function allows 1e-12.
    
    # Let's re-verify radii calculation
    n = centers_final.shape[0]
    radii_recalc = np.zeros(n)
    for i in range(n):
        r = min_distance_to_boundary(centers_final[i])
        for j in range(n):
            if i == j: continue
            d = distance(centers_final[i], centers_final[j])
            # Max radius such that circle i doesn't overlap j
            # dist >= r_i + r_j => r_i <= dist - r_j
            # But r_j is not fixed yet.
            # We want to find max r_i for all i.
            # This is a system.
            # However, for the "fixed centers" case, the max radius for i is limited by:
            # r_i <= dist(i,j) - r_j? No.
            # If centers are fixed, we can just set r_i = min(dist(i,j)/2, boundary_dist).
            # This guarantees r_i + r_j <= dist(i,j).
            # Because r_i <= d/2 and r_j <= d/2 => r_i + r_j <= d.
            # So the simple calculation is correct.
            if d/2 < r:
                r = d/2
        radii_recalc[i] = r
    
    radii_final = radii_recalc
    
    sum_radii = np.sum(radii_final)
    
    return centers_final, radii_final, float(sum_radii)

# To allow running the function
if __name__ == "__main__":
    # Just a check if run locally
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")