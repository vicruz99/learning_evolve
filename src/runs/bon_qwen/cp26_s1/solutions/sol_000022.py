# sol_000022 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6773994b) state=71e93680 sum of radii=2.447645 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal initialization and L-BFGS-B optimization with penalty method.
    """
    n_circles = 26
    
    # --- 1. Initialization: Hexagonal Grid ---
    # We want to generate roughly 26 points in a hexagonal pattern.
    # A triangular lattice has points at (i*dx, j*dy) and (i*dx + dx/2, j*dy + dy/2).
    # For a square, we can approximate this by placing points in rows.
    
    centers = np.zeros((n_circles, 2))
    idx = 0
    
    # Approximate grid size. sqrt(26) is approx 5.1.
    # Let's try a 5x5 grid plus one, or a hexagonal arrangement.
    # A simple hexagonal packing generation:
    # Rows with alternating shifts.
    
    # Let's estimate optimal radius r ~ 0.1. 
    # Spacing 2r ~ 0.2.
    # We can generate points and then let optimizer fix it.
    
    # Generate a dense grid of points and pick first 26?
    # Or construct specific rows.
    # 26 = 5 + 6 + 5 + 6 + 4 ? No, width constraint.
    # Let's just create a perturbed grid.
    
    # Create a 6x5 grid (30 points) and pick 26, or just place 26.
    # Let's place them in a grid first.
    rows = 6
    cols = 5 # 30 points
    # But we only need 26.
    
    # Let's try to arrange in hexagonal rows manually for better start.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 1 circle? 
    # 5*5 + 1 = 26.
    
    # Initial radius guess
    r_init = 0.08 # Safe small radius
    
    # We will optimize u and r.
    # u coordinates in [0,1].
    # Map u to x: x = r + u*(1-2r)
    # If r=0.08, 1-2r = 0.84.
    # x ranges from 0.08 to 0.92.
    
    # Initialize u to be spread out in [0,1]
    # Grid of u
    u_centers = np.zeros((n_circles, 2))
    count = 0
    
    # Hexagonal pattern in u-space (normalized)
    # We want points to be roughly evenly spaced.
    # Let's try to fill a square with hexagonal lattice.
    # Spacing in u should be approx 1/5 = 0.2
    
    step_u = 0.2
    for row in range(6):
        y_u = row * step_u
        # Shift every other row
        shift = step_u / 2 if row % 2 == 1 else 0
        # How many points in this row?
        # We need 26 points total.
        # Rows 0, 2, 4 (even): 5 points? x_u = 0.1, 0.3, 0.5, 0.7, 0.9
        # Rows 1, 3, 5 (odd): 5 points? x_u = 0.2, 0.4, 0.6, 0.8, 1.0?
        # Let's just generate points and stop at 26.
        
        x_u_start = shift
        # Determine number of points
        if row % 2 == 0:
             # 5 points: 0.1, 0.3, 0.5, 0.7, 0.9
             x_us = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        else:
             # 5 points: 0.2, 0.4, 0.6, 0.8, 1.0? 
             # Actually 1.0 is edge. Maybe 0.2, 0.4, 0.6, 0.8.
             # Let's do 4 or 5.
             # If we do 5 points shifted by 0.1: 0.2, 0.4, 0.6, 0.8, 1.0?
             # u=1.0 is allowed.
             x_us = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
             
        for xu in x_us:
            if count < n_circles:
                u_centers[count, 0] = xu
                u_centers[count, 1] = y_u
                count += 1
            else:
                break
        if count >= n_circles:
            break
            
    # Initialize radii
    radii = np.full(n_circles, r_init)
    
    # --- 2. Objective and Optimization ---
    
    # Variables: [u_0x, u_0y, ..., u_25x, u_25y, r_0, ..., r_25]
    # Size: 2*26 + 26 = 78
    # Bounds: u in [0, 1], r in [0, 0.5]
    
    def unpack(params):
        n = 26
        u = params[:2*n].reshape((n, 2))
        r = params[2*n:]
        return u, r

    def to_physical(u, r):
        # x = r + u * (1 - 2r)
        # y = r + u * (1 - 2r)
        # This ensures r <= x <= 1-r
        scale = 1.0 - 2.0 * r
        # scale can be negative if r > 0.5, but r bounded by 0.5.
        # If r=0.5, scale=0, x=0.5.
        centers = r[:, None] + u * scale[:, None]
        return centers

    def objective(params):
        u, r = unpack(params)
        centers = to_physical(u, r)
        
        # Sum of radii (we minimize negative sum)
        sum_r = np.sum(r)
        
        # Penalty for overlaps
        # We need to check all pairs (i, j) with i < j
        # Constraint: dist^2 >= (r_i + r_j)^2
        # Penalty: max(0, (r_i + r_j)^2 - dist^2)^2
        
        # Compute pairwise squared distances efficiently
        # centers shape (N, 2)
        # pdist computes condensed distance matrix
        dists = pdist(centers) # Euclidean distances
        
        # We need (r_i + r_j) for all pairs.
        # r shape (N,)
        # Create a matrix of sums
        # Using broadcasting might be easier or just loop if N is small.
        # N=26, pairs ~ 325. Loop is fine.
        
        penalty = 0.0
        # Vectorized pair check
        # indices of upper triangle
        i_indices, j_indices = np.triu_indices(n_circles, k=1)
        
        # r_i + r_j
        r_sums = r[i_indices] + r[j_indices]
        
        # dist^2
        dists_sq = dists ** 2
        
        # Violation
        violation = (r_sums ** 2) - dists_sq
        # Only penalize if violation > 0
        # penalty = sum(violation^2) where violation > 0
        # Using mask
        mask = violation > 0
        if np.any(mask):
            penalty = np.sum(violation[mask] ** 2)
            
        # Penalty weight
        # Needs to be large enough to enforce constraints
        # Since r ~ 0.1, r^2 ~ 0.01. Dist^2 ~ 0.04.
        # Violation could be around 0.01.
        # Penalty ~ 0.0001.
        # Sum r ~ 2.6.
        # We need penalty to dominate when violated.
        # Let's use a high weight.
        penalty_weight = 10000.0 
        
        return -sum_r + penalty_weight * penalty

    # Initial parameters
    init_params = np.concatenate([u_centers.flatten(), radii])
    
    # Bounds
    # u_x, u_y in [0, 1]
    # r in [0, 0.5] (actually max radius in square is 0.5)
    bounds = [(0, 1)] * (2 * n_circles) + [(0, 0.5)] * n_circles
    
    # Optimization
    # L-BFGS-B is good for bounds
    result = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-12})
    
    # Extract solution
    final_params = result.x
    u_sol, r_sol = unpack(final_params)
    centers_sol = to_physical(u_sol, r_sol)
    
    # --- 3. Post-processing ---
    # Ensure strict non-overlap and boundary compliance
    # The transformation handles boundary.
    # Check overlaps.
    
    # Validate and shrink if necessary
    # Compute min overlap margin
    dists = pdist(centers_sol)
    i_indices, j_indices = np.triu_indices(n_circles, k=1)
    r_sums = r_sol[i_indices] + r_sol[j_indices]
    
    # dist >= r_i + r_j  =>  dist - (r_i + r_j) >= 0
    margins = dists - r_sums
    
    min_margin = np.min(margins)
    
    # If min_margin < 0, we have overlap.
    # We need to shrink radii.
    # A simple way: scale all radii by factor f such that new margins >= 0.
    # However, shrinking radii changes dist? No, centers fixed.
    # dist is constant.
    # We need dist >= f*r_i + f*r_j = f*(r_i+r_j).
    # So f <= dist / (r_i + r_j).
    # For all pairs, f <= dist / (r_i + r_j).
    # So f = min(dists / r_sums).
    # But r_sums can be 0? Radii are positive.
    
    # Be careful with r_sums == 0 (if radius is 0).
    # Filter out pairs with 0 sum?
    # If r_i=0, no overlap constraint effectively (point).
    # But validation requires r >= 0.
    
    valid_pairs_mask = r_sums > 1e-9
    if np.any(valid_pairs_mask):
        ratios = np.ones_like(dists)
        ratios[valid_pairs_mask] = dists[valid_pairs_mask] / r_sums[valid_pairs_mask]
        min_ratio = np.min(ratios)
    else:
        min_ratio = 1.0
        
    # Apply shrinkage if needed
    # Allow small epsilon for numerical safety
    if min_ratio < 1.0 - 1e-9:
        scale_factor = min_ratio * (1.0 - 1e-6) # slight extra shrink
        r_sol = r_sol * scale_factor
        # Update centers to reflect new radii?
        # The transformation x = r + u(1-2r) depends on r.
        # If we change r, x changes.
        # But we just computed r_sol based on old r?
        # Wait, the penalty method optimized r and u together.
        # If we just scale r, we must recompute centers to stay valid?
        # Actually, if we scale r down, the circles get smaller, so overlaps disappear.
        # But the center positions might become invalid relative to new r?
        # Boundary check: r <= x <= 1-r.
        # If r decreases, the interval [r, 1-r] widens.
        # Since old x was in [old_r, 1-old_r], and new_r < old_r,
        # x is definitely in [new_r, 1-new_r].
        # So boundary is safe.
        # But we need to update centers?
        # The centers returned should be consistent with radii?
        # No, centers and radii are independent in the output, 
        # except for constraints.
        # But if we scaled r, we didn't change centers.
        # Is the pair (centers_sol, r_sol_scaled) valid?
        # Yes, because dists are same, r sums decreased, so dist > r_sum.
        # And boundary: x is same, r decreased, so x > r holds.
        
        # However, we might want to re-optimize or just return.
        # The objective sum of radii decreases, but it's a valid packing.
        # It might be suboptimal compared to a valid configuration found by optimizer.
        # But if optimizer failed to satisfy constraints exactly, this fixes it.
        
        # Let's recompute centers using the transformation with new r?
        # No, the transformation was just a parameterization for the optimizer.
        # The physical meaning is circle at center with radius r.
        # If we just output (centers_sol, r_sol_scaled), it is a valid packing.
        # But maybe we can do better by re-running optimizer or just accepting.
        # Given the complexity, fixing overlaps by shrinking is a safe fallback.
        
        # To maximize sum, maybe we should adjust centers?
        # But let's stick to simple correction.
        # Actually, the optimizer with high penalty should have found a valid point.
        # The penalty was 10000 * violation^2.
        # If violation was 1e-5, penalty 1e-6, negligible.
        # So solution should be very close to valid.
        
    # Recompute sum
    sum_radii = np.sum(r_sol)
    
    # Final Validation Check (mental check)
    # Ensure no NaNs
    if np.isnan(centers_sol).any() or np.isnan(r_sol).any():
        # Fallback to grid
        centers_sol = np.array([[i/6, j/4] for i in range(5) for j in range(5)] + [[0.5, 0.5]])[:26]
        r_sol = np.full(26, 0.05)
        sum_radii = np.sum(r_sol)

    return centers_sol, r_sol, sum_radii

# Helper to make it run if executed directly? 
# No, just define the function.
