import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    """
    n = 26
    
    # Strategy: Optimize for equal radii.
    # We use a force-directed simulation to push circles apart.
    
    # 1. Initialize centers
    # Start with a perturbed hexagonal grid which is a good lower bound.
    centers = np.zeros((n, 2))
    
    # Heuristic initialization: Hexagonal packing rows
    # Try to fit rows of 5 and 6 circles
    # Row lengths: 5, 6, 5, 6, 4 (sum=26) or similar
    # Let's try 5, 5, 5, 5, 6 (sum=26) - might be tight on width.
    # Let's try 5, 6, 5, 6, 4?
    # Let's just place them in a 5x5 grid plus one, then let optimizer fix it.
    # Or better: 6 rows.
    
    row_lengths = [5, 6, 5, 6, 4] # Sum = 26
    # This arrangement might be tall. Let's try to fit in 5 rows?
    # 6, 5, 5, 5, 5 = 26.
    row_lengths = [6, 5, 5, 5, 5]
    
    idx = 0
    y_pos = 0.1 # Start low, will scale
    row_height = 0.18 # Approx spacing
    
    # We will initialize loosely and let the optimizer tighten it.
    # Just place them in a grid for safety first.
    # 26 points. 6x5 grid is 30 points.
    # Let's take a subset of a grid.
    count = 0
    for i in range(6):
        for j in range(5):
            if count < n:
                centers[count, 0] = 0.2 + j * 0.2 # x: 0.2, 0.4, 0.6, 0.8, 1.0 (1.0 is bad)
                centers[count, 1] = 0.2 + i * 0.15 # y: 0.2, 0.35, ...
                # Adjust to keep inside
                centers[count, 0] = 0.1 + j * 0.25 # 0.1, 0.35, 0.6, 0.85, 1.1 (bad)
                # Let's use linspace
                pass
            count += 1

    # Better Initialization:
    # Place 26 points in a 5x6 grid pattern but only pick 26.
    # Or just a perturbed 5x5 grid.
    # Let's use a dense grid initialization.
    pts = []
    # 6 rows, 5 cols
    for i in range(6):
        for j in range(5):
            if len(pts) < n:
                # Hexagonal shift
                x = 0.1 + j * 0.2
                if i % 2 == 1:
                    x += 0.1 # Shift odd rows
                y = 0.1 + i * 0.18
                pts.append([x, y])
    
    centers = np.array(pts[:n])
    
    # 2. Optimization
    # We will iteratively increase a target radius r.
    # In each step, we run a few iterations of force-directed layout to resolve overlaps.
    
    current_r = 0.01
    max_r = 0.0 # Will track the best valid r found
    
    # We can try to ramp up r
    # A good upper bound estimate: Area ~ 1. 26 * pi * r^2 <= 1 -> r <= 0.11
    # But packing efficiency < 1. r approx 0.10.
    
    # Let's try to find the max r by binary search or just incremental search
    # Incremental search with local optimization is robust.
    
    r_target = 0.05
    step = 0.0001
    max_iterations = 2000
    learning_rate = 0.01
    
    # We will try to push r_target up until we can't satisfy constraints for a while.
    
    # Pre-calculate interaction pairs for speed
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j))
    pairs = np.array(pairs)
    
    best_centers = centers.copy()
    best_sum_r = 0.0
    
    # Let's try a fixed number of optimization steps for a range of r
    # Or just one long optimization where we slowly increase repulsion strength?
    
    # Approach: Simulate physics.
    # Repulsion force if dist < 2*r.
    # We want to find the largest r such that stable config exists.
    
    # Let's fix r and try to minimize energy.
    # If min energy < threshold, r is feasible.
    
    # Binary search for r
    low = 0.01
    high = 0.15
    feasible_r = low
    best_valid_centers = centers
    
    # Optimization function
    def optimize_for_r(centers_init, r, iterations=500, lr=0.005):
        c = centers_init.copy()
        for _ in range(iterations):
            forces = np.zeros_like(c)
            
            # Circle-Circle repulsion
            # Vectorized
            # dist matrix
            diff = c[:, np.newaxis, :] - c[np.newaxis, :, :] # (n, n, 2)
            dists = np.sqrt(np.sum(diff**2, axis=2)) # (n, n)
            np.fill_diagonal(dists, np.inf) # Ignore self
            
            # Find pairs with dist < 2r
            # We only need to process overlapping or close pairs
            # To be efficient, let's just process all and mask
            
            # Overlap amount
            overlap = np.maximum(0, 2*r - dists)
            
            # Force direction
            # If dist is 0, force is undefined, handle separately
            safe_dists = np.where(dists < 1e-9, 1e-9, dists)
            dir_vec = diff / safe_dists[:, :, np.newaxis] # (n, n, 2)
            
            # Force magnitude: proportional to overlap
            # F = overlap * dir
            # Sum forces for each node
            # Since matrix is symmetric, force on i from j is F_ij
            # Total force on i = sum_j F_ij
            # F_ij = overlap_ij * dir_ij (where dir_ij points from j to i? No, diff is c_i - c_j, so points j->i)
            # Yes, c_i - c_j points from j to i. Pushing i away from j.
            
            # Elementwise multiply
            f_mag = overlap[:, :, np.newaxis] # (n, n, 1)
            force_pairs = f_mag * dir_vec # (n, n, 2)
            
            # Sum over j axis (axis 1)
            forces += np.sum(force_pairs, axis=1)
            
            # Boundary repulsion
            # x bounds: r, 1-r
            # Force if x < r: push right (+)
            # Force if x > 1-r: push left (-)
            margin_x = np.maximum(0, r - c[:, 0]) - np.maximum(0, c[:, 0] - (1-r))
            margin_y = np.maximum(0, r - c[:, 1]) - np.maximum(0, c[:, 1] - (1-r))
            
            forces[:, 0] += margin_x
            forces[:, 1] += margin_y
            
            # Apply forces
            # Adaptive step size or fixed
            # To prevent explosion, cap force or use small lr
            c += lr * forces
            
            # Clamp to [0, 1] strictly to avoid runaway, though physics should keep them in
            c = np.clip(c, 0, 1)
            
        return c

    # Binary search
    for _ in range(50): # 50 iterations of binary search
        mid_r = (low + high) / 2
        # Random restart for optimization to avoid local minima
        # But we have a decent init.
        # Let's try multiple random shifts
        best_c = None
        min_energy = np.inf
        
        # Try a few random perturbations of the current best or init
        candidates = [best_valid_centers.copy()]
        # Add random noise candidates
        for _ in range(3):
            noise = np.random.uniform(-0.05, 0.05, (n, 2))
            cand = best_valid_centers + noise
            cand = np.clip(cand, 0, 1)
            candidates.append(cand)
            
        for c_init in candidates:
            c_opt = optimize_for_r(c_init, mid_r, iterations=300, lr=0.01)
            
            # Check energy (sum of overlaps squared + boundary violation squared)
            # If energy is low enough, it's feasible
            diff = c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            overlap = np.maximum(0, 2*mid_r - dists)
            overlap_sum = np.sum(overlap)
            
            # Boundary check
            bound_viol = np.maximum(0, mid_r - c_opt[:, 0]) + \
                         np.maximum(0, c_opt[:, 0] - (1-mid_r)) + \
                         np.maximum(0, mid_r - c_opt[:, 1]) + \
                         np.maximum(0, c_opt[:, 1] - (1-mid_r))
            bound_sum = np.sum(bound_viol)
            
            energy = overlap_sum + bound_sum
            
            if energy < min_energy:
                min_energy = energy
                best_c = c_opt
        
        if min_energy < 1e-6: # Feasible
            low = mid_r
            best_valid_centers = best_c
            feasible_r = mid_r
        else:
            high = mid_r
            
    final_r = feasible_r
    final_centers = best_valid_centers
    radii = np.full(n, final_r)
    sum_radii = np.sum(radii)
    
    # Final polish: Run optimization one last time with found r to clean up
    final_centers = optimize_for_r(final_centers, final_r, iterations=1000, lr=0.005)
    
    # Ensure strict validity (clamp if necessary, though logic should hold)
    # The optimization might leave tiny violations.
    # Let's verify and adjust r slightly down if needed.
    
    # Recalculate min distance to be safe
    diff = final_centers[:, np.newaxis, :] - final_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Boundary distance
    dist_bdry = np.minimum(
        np.minimum(final_centers[:, 0], 1 - final_centers[:, 0]),
        np.minimum(final_centers[:, 1], 1 - final_centers[:, 1])
    )
    min_bdry = np.min(dist_bdry)
    
    # The actual feasible radius for this configuration
    # r_max = min(min_dist/2, min_bdry)
    # We set radii to this value
    r_actual = min(min_dist / 2, min_bdry)
    
    # To avoid numerical issues with 1e-12 tolerance, we can shrink slightly
    # But the validator allows 1e-12.
    # Let's use r_actual.
    
    radii = np.full(n, r_actual)
    sum_radii = np.sum(radii)
    
    return final_centers, radii, sum_radii