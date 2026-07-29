# sol_000171 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 891ad575) state=917a2359 sum of radii=2.136521 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def run_packing():
    n = 26
    # 1. Initialization: Hexagonal Lattice
    # Estimate initial radius. 26 circles in unit square, hex packing density ~0.9069
    # Area ~ 1. N * pi * r^2 * density = 1 => r ~ sqrt(1 / (26 * pi * 0.9069)) ~ 0.105
    r_init = 0.105
    
    centers = []
    # Generate hex grid
    y = r_init
    row_idx = 0
    while y + r_init <= 1.0:
        x = r_init
        # Offset x for alternating rows
        x_offset = r_init if row_idx % 2 == 1 else 0
        x = r_init + x_offset
        
        while x + r_init <= 1.0:
            centers.append([x, y])
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row_idx += 1
    
    # If we don't have 26 points, pad with random points or grid points
    # The loop above might yield slightly fewer or more depending on r_init.
    # Let's ensure we have exactly 26.
    if len(centers) < n:
        # Fill remaining with random points inside [0.1, 0.9] to avoid immediate boundary issues
        while len(centers) < n:
            cx = np.random.uniform(0.1, 0.9)
            cy = np.random.uniform(0.1, 0.9)
            centers.append([cx, cy])
    else:
        # If we have more, trim. Prefer keeping the first ones (top-left density).
        centers = centers[:n]
        
    centers = np.array(centers)
    
    # 2. Optimize Centers for Equal Radii
    # We want to maximize r such that constraints are satisfied.
    # Equivalent to minimizing the penalty for overlaps and boundary violations for a target r.
    # However, r is unknown. We can optimize centers to maximize the minimum distance (maximin).
    # Let's define a function that returns the "minimum clearance" (approx radius) for a set of centers.
    # clearance = min(min_dist_to_boundary, min_dist_between_centers / 2)
    # We want to maximize this clearance.
    
    def clearance(centers_flat):
        c = centers_flat.reshape(-1, 2)
        min_d = 1.0
        
        # Boundary constraints
        # dist to boundary = min(x, 1-x, y, 1-y)
        d_boundary = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                                np.minimum(c[:, 1], 1.0 - c[:, 1]))
        min_d = min(min_d, np.min(d_boundary))
        
        # Inter-circle constraints
        # dist / 2
        # Compute pairwise distances
        # To save time, only check a subset or use broadcasting for N=26
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Lower triangle
        np.fill_diagonal(dists, 1.0) # Ignore self
        min_pair_dist = np.min(dists)
        min_d = min(min_d, min_pair_dist / 2.0)
        
        return min_d

    # We want to maximize clearance. So minimize -clearance.
    # Use bounds for centers [0, 1]
    bounds = [(0, 1)] * (n * 2)
    
    # Run optimization multiple times or use a robust method
    # L-BFGS-B is good for box constraints
    initial_flat = centers.flatten()
    
    # Note: The clearance function is non-smooth (min of distances).
    # We can use a smooth approximation or just rely on the optimizer handling the kinks.
    # For robustness, let's minimize the sum of squared violations for a target radius r,
    # but finding the optimal r is the goal.
    # Let's stick to maximizing the clearance directly.
    
    # To make it smoother, we can use a penalty method.
    # Objective: - clearance.
    # Let's try to optimize this.
    
    res_centers = None
    best_val = -1
    
    # Try a few random restarts or just one good run from hex grid
    # Hex grid is already quite good.
    
    # We can use a simple loop to iteratively improve.
    # But scipy minimize should work.
    
    def objective(x):
        return -clearance(x)
        
    try:
        res = minimize(objective, initial_flat, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6})
        res_centers = res.x.reshape(-1, 2)
        r_opt = clearance(res_centers)
    except Exception:
        res_centers = centers
        r_opt = clearance(res_centers)

    # 3. Refine with Unequal Radii using LP
    # For fixed centers res_centers, find radii r_i to maximize sum(r_i)
    # Constraints:
    # 1. r_i <= x_i
    # 2. r_i <= 1 - x_i
    # 3. r_i <= y_i
    # 4. r_i <= 1 - y_i
    # 5. r_i + r_j <= dist(i, j) for all i < j
    
    c = res_centers
    dist_matrix = np.sqrt(np.sum((c[:, np.newaxis, :] - c[np.newaxis, :, :])**2, axis=2))
    
    # LP formulation
    # Maximize sum(r) => Minimize -sum(r)
    # c_obj = -1 for all r_i
    c_obj = np.ones(n) * -1
    
    # A_ub @ r <= b_ub
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= limit
    limits = np.zeros((n, 4))
    limits[:, 0] = c[:, 0]           # x
    limits[:, 1] = 1.0 - c[:, 0]    # 1-x
    limits[:, 2] = c[:, 1]           # y
    limits[:, 3] = 1.0 - c[:, 1]    # 1-y
    
    min_limit = np.min(limits, axis=1)
    
    # Constraints r_i <= min_limit
    # I @ r <= min_limit
    A_ub_boundary = np.eye(n)
    b_ub_boundary = min_limit
    
    # Constraints r_i + r_j <= dist_ij
    # For each pair (i, j), row with 1 at i, 1 at j
    pair_constraints = []
    pair_rhs = []
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            pair_constraints.append(row)
            pair_rhs.append(dist_matrix[i, j])
            
    A_ub_pairs = np.array(pair_constraints)
    b_ub_pairs = np.array(pair_rhs)
    
    # Combine
    A_ub_total = np.vstack([A_ub_boundary, A_ub_pairs])
    b_ub_total = np.concatenate([b_ub_boundary, b_ub_pairs])
    
    # Bounds for r: r_i >= 0
    bounds_r = [(0, None)] * n
    
    try:
        res_lp = linprog(c_obj, A_ub=A_ub_total, b_ub=b_ub_total, bounds=bounds_r, method='highs')
        if res_lp.success:
            radii = res_lp.x
            centers_final = c
            sum_radii = np.sum(radii)
            # Validation check just in case (numerical noise)
            # The LP ensures constraints are met within tolerance.
        else:
            # Fallback to equal radii from optimization
            radii = np.full(n, r_opt)
            centers_final = c
            sum_radii = 26 * r_opt
    except Exception:
        radii = np.full(n, r_opt)
        centers_final = c
        sum_radii = 26 * r_opt

    # Final sanity clamp to ensure no negative radii or tiny violations
    # (LP should handle this, but good for robustness)
    
    return centers_final, radii, float(sum_radii)
