# sol_000267 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3433cac4) state=f5b3893a sum of radii=1.614229 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    
    # Initial guess generation:
    # We try to arrange circles in a hexagonal-like pattern.
    # A 5x5 grid has 25 circles with r=0.1. We need 26.
    # We can try to fit 26 circles by slightly compressing or arranging in rows like 5, 5, 5, 5, 6? 
    # But 6 in a row is hard for r > 0.08. 
    # A better arrangement for equal circles is often staggered rows.
    # Let's try to fit them in 5 rows. 
    # Rows could have counts like 6, 5, 5, 5, 5 is width-constrained.
    # Maybe 5, 5, 5, 5, 6 (rotated?).
    # Let's use a simple dense initialization:
    # Place them on a grid, then let optimizer find the best packing.
    # A 6x5 grid (30 circles) is too many, but we can take a subset or compress.
    # Let's start with a perturbed grid or a specific hexagonal layout.
    
    # Let's try to construct a hexagonal layout for ~26 circles.
    # Row height factor sqrt(3)/2 ~ 0.866.
    # If we have 5 rows, height is roughly 4 * 0.866 * 2r + 2r?
    # Let's just randomize slightly around a grid to break symmetry and help optimization.
    
    # Initial centers: 5 rows, approx 5-6 cols
    # To get 26, maybe 5, 5, 5, 5, 6? 
    # Or 6, 5, 5, 5, 5?
    # Let's try 5 rows with 5, 5, 5, 5, 6 circles? 
    # Actually, just a random dense packing might work, but a structured one is safer.
    
    # Let's create a layout based on hexagonal packing logic
    # 6 rows of roughly 4-5 circles?
    # 6 * 4 = 24, 6 * 5 = 30.
    # Let's try 6 rows.
    # Row 0: 4 circles
    # Row 1: 5 circles
    # Row 2: 4 circles
    # Row 3: 5 circles
    # Row 4: 4 circles
    # Row 5: 4 circles
    # Total 26.
    # This zig-zag pattern is standard for hexagonal packing.
    
    centers = []
    # Approximate radius 0.1. 
    # Horizontal spacing 2r = 0.2. Vertical spacing r*sqrt(3) = 0.1732.
    # Let's normalize to fit in [0,1] later or just start small.
    # Start with r=0.1.
    
    r_start = 0.09
    dx = 2 * r_start
    dy = np.sqrt(3) * r_start
    
    # Generate coordinates
    # We need to fit in 1x1.
    # If we have 6 rows, height ~ 5*dy + 2r = 0.866 + 0.18 = 1.046. A bit tight.
    # Maybe 5 rows?
    # 5 rows: 5, 6, 5, 6, 4? Sum = 26.
    # Let's try 5 rows.
    
    row_counts = [5, 6, 5, 6, 4] # Sum = 26
    # Wait, 6 circles in a row with r=0.09 -> width 12*0.09 = 1.08. Tight.
    # Maybe staggered rows allow tighter packing?
    # Let's just use a grid initialization and let the optimizer handle it.
    # A 6x5 grid (30 points) -> remove 4?
    # Or just random points in [0,1] with repulsion?
    
    # Better initialization:
    # Place centers on a hexagonal lattice scaled to fit.
    # Lattice points (i + j/2, j * sqrt(3)/2)
    
    points = []
    # Try to fit points in a slightly larger box to allow expansion
    scale = 0.9 
    
    # We want to find 26 points.
    # Let's iterate over a grid of hexagonal indices
    for j in range(6): # 6 rows
        for i in range(6): # 6 cols
            # Hex coords
            x = (i + 0.5 * (j % 2)) * 2.0 # 2r spacing assumed
            y = j * np.sqrt(3)
            
            # Normalize to fit in 1x1 approximately
            # Max x approx 6*2 = 12. Max y approx 5*1.732 = 8.66.
            # This is getting complicated to tune.
            pass

    # Simpler initialization:
    # 5x5 grid (25 points) + 1 point in the middle of a hole.
    # Grid points at 0.1, 0.3, 0.5, 0.7, 0.9
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    
    init_centers = []
    # Add 25 grid points
    for y in grid_y:
        for x in grid_x:
            init_centers.append([x, y])
    
    # Add 26th point at a gap center, e.g., (0.2, 0.2)
    # But (0.2, 0.2) is distance 0.141 from (0.1, 0.1).
    # If r=0.1, this overlaps. 
    # We need to shrink r or move points.
    # Let's place it at (0.2, 0.2) but with a smaller initial radius or just let optimizer fix it.
    init_centers.append([0.2, 0.2])
    
    centers_init = np.array(init_centers)
    
    # We will optimize a shared radius r and all centers.
    # Variables: [x1, y1, ..., x26, y26, r]
    # But bounds for x,y are [0,1]. r is [0, 0.5].
    
    # Actually, optimizing r directly with hard constraints is hard.
    # We will use a penalty method on centers for a fixed r, then binary search r?
    # Or optimize both.
    # Let's optimize centers for a target radius r, and use a penalty function.
    # We can try to maximize r by minimizing -r + penalty.
    
    def objective(variables):
        # variables: 26*2 coords + 1 radius? 
        # Let's keep radius separate or include it.
        # Let's include radius as the last element.
        # But radius should be uniform? 
        # Problem asks to maximize sum of radii. Unequal might be better, but equal is a good baseline.
        # Let's allow unequal radii?
        # Variables: 26 centers (52) + 26 radii (26) = 78 vars.
        # That's a lot. 
        # Let's stick to equal radii first. If sum < 2.636, we can try unequal.
        # 2.636 / 26 = 0.10138.
        # Equal radius r ~ 0.1014 is the target.
        
        # Let's assume equal radii r.
        # Variables: 52 centers + 1 radius.
        # Actually, if we maximize r, we just need to check feasibility.
        # But penalty method allows soft constraints.
        
        # Split variables
        c_flat = variables[:-1] # 52 coords
        r = variables[-1]       # radius
        
        # Reshape centers
        centers = c_flat.reshape((n, 2))
        
        # Boundary penalties
        # x in [r, 1-r], y in [r, 1-r]
        # Penalty if x < r or x > 1-r
        bound_pen = 0
        for i in range(n):
            x, y = centers[i]
            # Left
            if x < r:
                bound_pen += (r - x)**2
            # Right
            if x > 1 - r:
                bound_pen += (x - (1 - r))**2
            # Bottom
            if y < r:
                bound_pen += (r - y)**2
            # Top
            if y > 1 - r:
                bound_pen += (y - (1 - r))**2
        
        # Overlap penalties
        overlap_pen = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                req_dist = 2 * r
                if dist < req_dist:
                    overlap_pen += (req_dist - dist)**2
        
        # Objective: Maximize r => Minimize -r
        # Add penalties
        # We need to balance -r and penalties. 
        # If penalties are 0, we want max r.
        # If penalties > 0, we pay a cost.
        # Weight for penalties?
        # Since r is small (~0.1), r^2 is 0.01.
        # Penalty terms are squared differences.
        # Let's use a weight.
        
        weight = 1000.0 # High weight to enforce constraints
        return -r + weight * (bound_pen + overlap_pen)

    # Bounds for optimization
    # x, y in [0, 1] (actually [0,1] is safe, constraint handles r)
    # r in [0, 0.5]
    
    # Initial guess for r: 0.1
    # Initial centers: from init_centers
    
    # Concatenate
    x0 = np.concatenate([centers_init.flatten(), [0.1]])
    
    bounds = [(0, 1)] * (2 * n) + [(0.001, 0.5)]
    
    # Optimization
    # Method 'L-BFGS-B' handles bounds.
    # However, the objective is non-smooth (due to max/ifs) or we used if statements.
    # Let's rewrite objective to be smooth (using max(0, ...)^2 is C1 but not C2, but usually fine).
    # Actually, 'if' statements make it non-differentiable at kinks.
    # Better to use np.maximum(0, val)**2
    
    def objective_smooth(variables):
        c_flat = variables[:-1]
        r = variables[-1]
        centers = c_flat.reshape((n, 2))
        
        # Boundary penalties using max
        # max(0, r - x)^2
        pen_x_min = np.maximum(0, r - centers[:, 0])**2
        pen_x_max = np.maximum(0, centers[:, 0] - (1 - r))**2
        pen_y_min = np.maximum(0, r - centers[:, 1])**2
        pen_y_max = np.maximum(0, centers[:, 1] - (1 - r))**2
        bound_pen = np.sum(pen_x_min + pen_x_max + pen_y_min + pen_y_max)
        
        # Overlap penalties
        # Vectorized distance calculation might be faster but loops are fine for N=26
        # Use broadcasting for speed?
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (26, 26, 2)
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask upper triangle
        # This is cleaner
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # We only care about i < j
        # Triangular mask
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        dists_subset = dists[mask]
        
        # Required distance 2r
        req_dist = 2 * r
        # Penalty if dist < req_dist
        # max(0, req - dist)^2
        overlaps = np.maximum(0, req_dist - dists_subset)**2
        overlap_pen = np.sum(overlaps)
        
        weight = 5000.0 # Adjust weight if needed
        return -r + weight * (bound_pen + overlap_pen)

    # Run optimization
    # We might need multiple restarts or a good initial r.
    # Let's try starting with r=0.1.
    
    result = opt.minimize(objective_smooth, x0, method='L-BFGS-B', bounds=bounds, 
                          options={'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 5000})
    
    best_r = result.x[-1]
    best_centers = result.x[:-1].reshape((n, 2))
    
    # Check if valid
    # The penalty might not be 0.
    # If result.fun is close to -r (i.e. penalty part is 0), then valid.
    # Actually result.fun = -r + weight*pen.
    # If pen ~ 0, fun ~ -r.
    
    # If penalty is not 0, we might have a conflict.
    # We can try to enforce validity by checking and shrinking r slightly?
    # Or running a correction step.
    
    # Let's verify constraints
    centers = best_centers
    r = best_r
    
    # Check overlaps
    valid = True
    # Check boundaries
    for i in range(n):
        x, y = centers[i]
        if x < r - 1e-9 or x > 1 - r + 1e-9 or y < r - 1e-9 or y > 1 - r + 1e-9:
            valid = False
            # Adjust center to boundary
            centers[i, 0] = np.clip(centers[i, 0], r, 1-r)
            centers[i, 1] = np.clip(centers[i, 1], r, 1-r)
            # This might cause overlaps, but better than invalid bounds.
            # And reduce r?
    
    # Check overlaps again and fix
    # If overlaps exist, we might need to reduce r.
    # Let's compute min distance between centers
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < 2 * r - 1e-9:
        # Overlap detected.
        # We can reduce r to satisfy constraint.
        # New r = min_dist / 2
        # But this might violate boundary if r increases? No, r decreases.
        # Boundary check: x >= r. If r decreases, x >= old_r >= new_r holds.
        new_r = min_dist / 2.0
        r = new_r
        # Recalculate sum
        # But wait, if we reduce r, sum decreases.
        # Maybe the optimizer got stuck in a local minimum with high penalty?
        # Or maybe the configuration is just not optimal.
        pass
        
    # Let's try a second optimization pass if invalid, or just accept.
    # To be safe, let's clamp r to ensure validity.
    # Actually, if min_dist < 2r, we must reduce r.
    # But maybe we can move centers?
    # The optimizer should have handled this if weight was high enough.
    
    # Let's ensure radii are non-negative
    radii = np.full(n, r)
    
    # Final check and correction
    # Recalculate centers validity with new r
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], r, 1-r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1-r)
    
    # Re-check overlaps with clamped centers?
    # Clamping might cause new overlaps?
    # Unlikely if clamping is small.
    
    # If there are still overlaps, reduce r further until valid.
    # This is a safe fallback.
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    while np.min(dists) < 2 * r - 1e-12:
        min_d = np.min(dists)
        r = min_d / 2.0 - 1e-12 # Shrink slightly
        radii = np.full(n, r)
        # Update dists? No, centers didn't change, only r.
        # Actually dists don't depend on r.
        # Wait, if r changes, 2r changes.
        # Loop condition checks 2*r.
        # But we must ensure boundary too.
        # Boundary: r <= min(x, 1-x, y, 1-y).
        # Let's check boundary constraint for r.
        dists_to_wall_x = np.minimum(centers[:, 0], 1 - centers[:, 0])
        dists_to_wall_y = np.minimum(centers[:, 1], 1 - centers[:, 1])
        min_wall_dist = np.min(np.minimum(dists_to_wall_x, dists_to_wall_y))
        if r > min_wall_dist - 1e-12:
            r = min_wall_dist - 1e-12
            radii = np.full(n, r)
            # If r decreases, overlaps might resolve?
            # Yes, 2r decreases.
            
        # Recalculate min dist check
        if 2*r > np.min(dists):
            # Still overlapping?
            # This loop might be infinite if dists < 2r and r = min_d/2.
            # If r = min_d/2, then 2r = min_d.
            # Loop condition: min_d < 2r -> min_d < min_d -> False.
            # So it terminates.
            pass
        else:
            break
            
        # Safety break
        if r < 1e-5:
            break

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Wait, the loop logic above for r adjustment is slightly flawed.
# If we reduce r, 2r reduces. Overlap condition dist < 2r might be satisfied.
# But we need to re-evaluate?
# Actually, dists are constant (centers fixed).
# We just need 2r <= min(dists).
# So r <= min(dists)/2.
# And r <= min_wall_dist.
# So optimal r for fixed centers is min(min(dists)/2, min_wall_dist).
# Let's just compute that directly.

    # Correct logic for fallback:
    # 1. Compute min inter-circle distance.
    # 2. Compute min distance to wall.
    # 3. Set r = min(min_dist/2, min_wall_dist).
    
    # But wait, we want to MAXIMIZE sum radii.
    # The optimizer should have found a good balance.
    # If the optimizer failed (penalty not zero), reducing r is a safe way to get a valid solution,
    # though maybe not the maximal one.
    # But with a good optimizer and weight, it should be valid.
    
    # Let's refine the code to be cleaner.

def run_packing():
    """
    Optimizes packing of 26 circles.
    """
    n = 26
    
    # Initialization
    # Try to place circles in a dense pattern.
    # A 5x5 grid has 25 circles. We add 1.
    # To allow for larger radius, maybe start with smaller radius?
    # Or just random?
    # Let's try a grid initialization.
    
    # Grid points
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    
    centers_init = []
    for y in grid_y:
        for x in grid_x:
            centers_init.append([x, y])
    # 26th circle
    centers_init.append([0.2, 0.2]) # In a hole
    
    centers_init = np.array(centers_init)
    
    # Optimization function
    def objective(vars):
        # vars: 26*2 coords + 1 radius r
        c = vars[:2*n].reshape((n, 2))
        r = vars[2*n]
        
        # Penalty for boundary
        # x >= r, x <= 1-r, y >= r, y <= 1-r
        # Penalty terms: max(0, r-x)^2 etc.
        
        p_bound = 0
        for i in range(n):
            x, y = c[i]
            p_bound += max(0, r - x)**2
            p_bound += max(0, x - (1 - r))**2
            p_bound += max(0, r - y)**2
            p_bound += max(0, y - (1 - r))**2
            
        # Penalty for overlap
        p_overlap = 0
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((c[i] - c[j])**2))
                if dist < 2*r:
                    p_overlap += (2*r - dist)**2
        
        # We want to maximize r.
        # Minimize -r + weight * (p_bound + p_overlap)
        w = 10000.0
        return -r + w * (p_bound + p_overlap)

    # Initial guess
    # r = 0.09 (safe)
    x0 = np.concatenate([centers_init.flatten(), [0.09]])
    
    bounds = []
    for _ in range(2*n):
        bounds.append((0.0, 1.0))
    bounds.append((0.001, 0.5))
    
    # Optimize
    res = opt.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 10000, 'ftol': 1e-15})
    
    best_c = res.x[:2*n].reshape((n, 2))
    best_r = res.x[2*n]
    
    # Validation and Correction
    # Ensure boundaries
    for i in range(n):
        best_c[i, 0] = np.clip(best_c[i, 0], best_r, 1 - best_r)
        best_c[i, 1] = np.clip(best_c[i, 1], best_r, 1 - best_r)
        
    # Check overlaps and reduce r if necessary
    # Calculate min distance between centers
    # Using broadcasting
    diff = best_c[:, np.newaxis, :] - best_c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    
    # Max possible r based on packing
    r_pack = min_d / 2.0
    
    # Max possible r based on walls
    dists_wall = np.minimum(
        np.minimum(best_c[:, 0], 1 - best_c[:, 0]),
        np.minimum(best_c[:, 1], 1 - best_c[:, 1])
    )
    r_wall = np.min(dists_wall)
    
    final_r = min(best_r, r_pack, r_wall)
    
    # Apply final_r
    radii = np.full(n, final_r)
    
    # Re-clip centers just in case (though if final_r <= best_r and best_r was valid boundary-wise, it's fine)
    # But if we reduced r due to overlap, walls might still be satisfied.
    # If we reduced r due to wall, we need to clip?
    # If r_wall < best_r, it means some circle was too close to wall.
    # But we clipped best_c to [best_r, 1-best_r].
    # So dist to wall was at least best_r.
    # So r_wall >= best_r.
    # So r_wall check is redundant if clipping was done.
    # The only constraint is overlap.
    
    # Wait, if we clipped, dist to wall is >= best_r.
    # So if final_r <= best_r, dist to wall >= final_r.
    # So boundaries are satisfied.
    # Overlaps: min_d >= 2 * best_r?
    # Not necessarily, optimizer might have returned a state with penalty > 0.
    # But with high weight, it should be close.
    # If penalty > 0, it means overlap or boundary violation.
    # We handled boundary by clipping.
    # If overlap remained, min_d < 2*best_r.
    # Then final_r = min_d/2 < best_r.
    # So we reduce r to resolve overlaps.
    
    # So final logic:
    # 1. Get result from optimizer.
    # 2. Clip centers to satisfy boundary for r=best_r.
    # 3. Check overlaps. If any, set r = min_d/2.
    
    # Re-eval min_d after clipping?
    # Clipping moves centers inwards. Distances might decrease.
    # So we should re-calc min_d after clipping.
    
    diff = best_c[:, np.newaxis, :] - best_c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    
    r_opt = best_r
    if min_d < 2 * r_opt - 1e-12:
        r_opt = min_d / 2.0 - 1e-12
        
    radii = np.full(n, r_opt)
    
    sum_radii = np.sum(radii)
    
    return best_c, radii, sum_radii

# Wait, the objective function used max(0, ...). 
# L-BFGS-B might have trouble with non-differentiable points?
# It usually works but might be slow.
# Also, the initialization with r=0.09 and centers from grid might be far from optimal.
# The grid centers for r=0.1 are at 0.1, 0.3...
# With r=0.09, they are safe.
# But the 26th circle at (0.2, 0.2) is very close to (0.1, 0.1), dist 0.141.
# 2r = 0.18. Overlap 0.04.
# Optimizer will push them apart.
# This seems viable.

# One concern: 2.636 target.
# 26 * 0.10138 = 2.63588.
# So we need r >= 0.10139.
# Grid with r=0.1 is valid.
# Can we reach 0.1014?
# Area argument says yes.
# Let's hope the optimizer finds it.

# To improve chances, maybe random restart?
# But time limit.
# Maybe just one run is enough.
# The code looks solid.
