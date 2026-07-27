# sol_000102 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bf51a1cd) state=73ab04aa sum of radii=1.956297 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def generate_hexagonal_initial_guess(n=26):
    """
    Generates an initial guess for centers based on a hexagonal packing pattern.
    Tries to distribute n circles into rows.
    """
    centers = np.zeros((n, 2))
    
    # Determine number of rows and circles per row to sum to n
    # A common dense pattern alternates row counts, e.g., 6, 5, 6, 5, 4 sums to 26.
    # Or 5, 5, 5, 5, 6?
    # Let's try to fit into a rectangle first.
    # Approx radius 0.1, diameter 0.2. 5x5 grid is 25.
    # We need 1 more.
    
    # Let's try 5 rows.
    # Rows counts: 6, 5, 6, 5, 4 -> sum 26.
    row_counts = [6, 5, 6, 5, 4]
    assert sum(row_counts) == n, "Row counts must sum to n"
    
    num_rows = len(row_counts)
    # Vertical spacing for hex packing: sqrt(3)/2 * diameter. 
    # Assuming diameter ~ 1/5 = 0.2, spacing ~ 0.173.
    # Total height ~ 5 * spacing.
    
    # We'll place them in [0, 1] x [0, 1] roughly.
    # y coordinates: evenly spaced between 0.1 and 0.9?
    # Actually, let's just place them and let optimizer fix it.
    
    # Hexagonal offset: row k is shifted by diameter/2 relative to row k-1.
    # Let's use a generic spacing scale.
    scale = 1.0 / 5.0 # roughly
    
    idx = 0
    for r_idx, count in enumerate(row_counts):
        # y coordinate for this row
        # Distribute rows evenly in y
        y = (r_idx + 0.5) / num_rows
        
        # x coordinates
        # If even row (0, 2...), start at some offset, if odd (1, 3...), shifted
        # Standard hex packing: centers of touching circles form equilateral triangles.
        # Horizontal dist = diameter. Vertical dist = sqrt(3)/2 * diameter.
        # Here we just need a good initial guess.
        
        # Shift for odd rows
        shift = 0.0
        if r_idx % 2 == 1:
            shift = 0.5 * scale # half step shift
        
        # Generate x coords
        # We want to fit 'count' circles in width 1.
        # Spacing between centers = 1 / (count - 1)? No, that's for touching walls.
        # Let's just space them evenly.
        # Range of x: [shift * scale, 1 - shift * scale]?
        # Actually, let's just put them in [0.1, 0.9] range initially.
        
        x_start = 0.1
        x_end = 0.9
        if count > 1:
            step = (x_end - x_start) / (count - 1)
        else:
            step = 0
            
        for c_idx in range(count):
            x = x_start + c_idx * step + (shift * step * 2) # rough shift
            # Clamp
            x = np.clip(x, 0.01, 0.99)
            y = np.clip(y, 0.01, 0.99)
            centers[idx] = [x, y]
            idx += 1
            
    return centers

def objective_equal_radius(params, n):
    """
    Objective function to maximize the minimum clearance.
    We want to maximize r such that circles of radius r fit.
    Constraints:
    1. r <= dist(center, boundary)
    2. 2r <= dist(center_i, center_j)
    
    We transform this to minimizing -r.
    But r is determined by the tightest constraint.
    Instead of passing r as a variable (which makes it non-convex coupled),
    we can just optimize centers to maximize min(dist_boundary, dist_pair/2).
    
    Actually, let's include r as a variable to be maximized?
    Or just return the min distance found, and the optimizer will try to increase it?
    No, the optimizer minimizes. So we minimize -min_distance.
    
    However, the function min_distance is not smooth.
    But Nelder-Mead handles it.
    """
    centers = params.reshape((n, 2))
    
    min_val = 1.0
    
    # Check boundary distances
    for i in range(n):
        x, y = centers[i]
        dist_boundary = min(x, 1.0 - x, y, 1.0 - y)
        if dist_boundary < min_val:
            min_val = dist_boundary
            
    # Check pairwise distances
    # Optimization: only check pairs, but O(N^2) is small for N=26 (325 pairs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            half_dist = dist / 2.0
            if half_dist < min_val:
                min_val = half_dist
                
    # We want to maximize min_val, so minimize -min_val
    return -min_val

def solve_packing():
    n = 26
    
    # 1. Initial Guess
    # Try a few different initializations and pick the best
    best_params = None
    best_val = -1.0
    
    # Option 1: Hexagonal-like
    centers1 = generate_hexagonal_initial_guess(n)
    params1 = centers1.flatten()
    val1 = objective_equal_radius(params1, n)
    
    # Option 2: Random uniform (good for diversity)
    np.random.seed(42)
    centers2 = np.random.rand(n, 2) * 0.8 + 0.1 # Keep inside [0.1, 0.9]
    params2 = centers2.flatten()
    val2 = objective_equal_radius(params2, n)
    
    # Option 3: Grid
    centers3 = np.zeros((n, 2))
    # 5x5 grid is 25 points. Add 1 in middle?
    # Just fill a 6x6 grid and take first 26
    grid_size = 6
    step = 1.0 / (grid_size + 1)
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            if idx < n:
                centers3[idx] = [(c + 1) * step, (r + 1) * step]
                idx += 1
    params3 = centers3.flatten()
    val3 = objective_equal_radius(params3, n)
    
    # Pick best initial
    candidates = [(params1, val1), (params2, val2), (params3, val3)]
    candidates.sort(key=lambda x: x[1], reverse=True) # Higher val (less negative) is better? 
    # Wait, val is negative of radius. So -0.1 is better than -0.05.
    # Wait, min_val is radius. objective returns -min_val.
    # So we want to minimize objective (make it more negative).
    # So -0.1 < -0.05.
    # So we want smallest objective value.
    
    candidates.sort(key=lambda x: x[1]) 
    best_params = candidates[0][0]
    best_obj = candidates[0][1]
    
    # 2. Optimization
    # Use Nelder-Mead as it doesn't require gradients and handles non-smoothness
    # Bounds: centers must be in [0, 1]. 
    # Nelder-Mead doesn't support bounds directly. 
    # We can use a penalty or just rely on the fact that optimal is inside.
    # But it might wander outside.
    # Let's use L-BFGS-B which supports bounds, but might get stuck on non-smooth function.
    # Let's try Nelder-Mead first with bounds enforcement inside objective?
    # Or just let it run. If it goes outside, distance to boundary becomes negative, 
    # which is "bad" (large positive objective? No, min_val becomes negative, -min_val positive).
    # So it will be penalized.
    
    bounds = [(0.0, 1.0) for _ in range(2 * n)]
    
    # Try Nelder-Mead
    res1 = opt.minimize(
        objective_equal_radius,
        best_params,
        args=(n,),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6}
    )
    
    # Try L-BFGS-B with bounds
    try:
        res2 = opt.minimize(
            objective_equal_radius,
            best_params,
            args=(n,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 10000, 'ftol': 1e-8}
        )
    except:
        res2 = res1
        
    # Select best result
    if res2.fun < res1.fun:
        best_result = res2
    else:
        best_result = res1
        
    best_centers = best_result.x.reshape((n, 2))
    max_radius = -best_result.fun
    
    # 3. Verify and fix any minor violations due to numerical error
    # Although the optimizer maximizes the minimum distance, 
    # the calculated max_radius might be slightly optimistic if we just take -fun.
    # We should clamp the radius to be valid.
    
    # Recalculate actual max valid radius for these centers
    actual_r = 1.0
    for i in range(n):
        x, y = best_centers[i]
        d_bound = min(x, 1.0 - x, y, 1.0 - y)
        if d_bound < actual_r:
            actual_r = d_bound
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            if d / 2.0 < actual_r:
                actual_r = d / 2.0
                
    # To be safe, reduce slightly
    actual_r = actual_r * (1.0 - 1e-5)
    
    radii = np.full(n, actual_r)
    sum_radii = np.sum(radii)
    
    # Check if sum is good enough.
    # Target 2.636.
    # If equal radii strategy fails to reach high enough, we might need unequal.
    # But let's assume equal is close.
    # If sum_radii < 2.6, maybe try unequal optimization?
    # But for now, return this.
    
    # Just to be robust, let's ensure we return valid radii.
    # The optimization maximizes the *equal* radius.
    # Is it possible that unequal radii give higher sum?
    # Yes.
    # Let's implement a quick local search to adjust radii individually to increase sum.
    # This is a linear programming problem for fixed centers.
    # But since we have the centers, we can just solve for max sum radii.
    
    # Solve LP: max sum(r_i)
    # s.t. r_i <= dist(c_i, boundary)
    # r_i + r_j <= dist(c_i, c_j)
    # r_i >= 0
    
    # We can use scipy.optimize.linprog
    # Variables: r_0, ..., r_25
    # Maximize sum(r) => Minimize -sum(r)
    # c = [-1, -1, ..., -1]
    
    # Constraints:
    # 1. r_i <= B_i  => r_i - B_i <= 0 ? No. r_i <= B_i.
    #    In linprog, A_ub @ x <= b_ub.
    #    x = [r_0, ...].
    #    Constraint: r_i <= B_i  =>  1*r_i <= B_i.
    #    Constraint: r_i + r_j <= D_ij => 1*r_i + 1*r_j <= D_ij.
    
    c_obj = np.ones(n) * -1 # Minimize -sum(r)
    
    # Upper bounds for r_i from walls
    # Actually linprog handles bounds on variables.
    # r_i >= 0 is default lower bound.
    # Upper bound for r_i is min(x_i, 1-x_i, y_i, 1-y_i).
    bounds_r = []
    upper_bounds_r = np.zeros(n)
    for i in range(n):
        x, y = best_centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        upper_bounds_r[i] = ub
        bounds_r.append((0, ub))
        
    # Pairwise constraints: r_i + r_j <= dist_ij
    A_ub = []
    b_ub = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(best_centers[i] - best_centers[j])
            # r_i + r_j <= dist
            # Row in A_ub: 0...1 at i...1 at j...0
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    if len(A_ub) > 0:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = np.array([]).reshape(0, n)
        b_ub = np.array([])

    try:
        res_lp = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res_lp.success:
            radii = res_lp.x
            # Clean up negative zeros
            radii = np.maximum(radii, 0.0)
            sum_radii = np.sum(radii)
            # Update centers? No, centers are fixed from previous step.
            # But maybe we can improve centers based on these radii?
            # Iterating might be complex. Let's stick to these radii.
            
            # One check: validate
            # We assume it's valid by construction.
    except Exception as e:
        # Fallback to equal radii
        radii = np.full(n, actual_r)
        sum_radii = np.sum(radii)

    return best_centers, radii, sum_radii

def run_packing():
    centers, radii, sum_radii = solve_packing()
    return centers, radii, sum_radii

if __name__ == "__main__":
    # Run validation if needed
    import numpy as np
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic check
    for i in range(len(centers)):
        x, y = centers[i]
        r = radii[i]
        if x-r < -1e-9 or x+r > 1+1e-9 or y-r < -1e-9 or y+r > 1+1e-9:
            print("Boundary violation")
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            d = np.sqrt(np.sum((centers[i]-centers[j])**2))
            if d < radii[i] + radii[j] - 1e-9:
                print(f"Overlap {i} {j}")
