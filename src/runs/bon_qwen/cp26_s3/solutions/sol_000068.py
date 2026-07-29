# sol_000068 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 16623584) state=adaf11bf sum of radii=0.546051 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def solve_radii_lp(centers):
    """
    Solves for the optimal radii given fixed centers using Linear Programming.
    Maximize sum(r_i) subject to non-overlap and boundary constraints.
    """
    n = centers.shape[0]
    # Bounds for variables r_i: 0 <= r_i <= 1 (loose upper bound)
    bounds = [(0, 1) for _ in range(n)]
    
    # Inequality constraints A_ub * r <= b_ub
    
    # 1. Boundary constraints:
    # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # Can be written as r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    
    # 2. Pairwise constraints:
    # r_i + r_j <= distance_ij
    # Construct matrix rows for these
    
    constraints_list = []
    
    # Precompute distances
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    
    # We will build the constraint matrix row by row to save memory/time if n was huge,
    # but for n=26, dense matrix is fine.
    # Total constraints: 4*n (boundaries) + n*(n-1)/2 (pairs)
    
    # Let's construct A_ub and b_ub
    # We only really need r_i <= boundary_limit for the LP to be tight.
    # And r_i + r_j <= dist_ij.
    
    # Actually, standard LP form is A_ub x <= b_ub.
    # For r_i <= limit, row is unit vector e_i, b is limit.
    # For r_i + r_j <= dist, row is e_i + e_j, b is dist.
    
    # Since scipy.linprog expects dense or sparse, let's build it.
    # 26 vars is small.
    
    # Boundary constraints
    A_ub = []
    b_ub = []
    
    for i in range(n):
        x, y = centers[i]
        limit = min(x, 1-x, y, 1-y)
        # r_i <= limit
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(limit)
        
    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = dists[i, j]
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    # Solve
    try:
        res = scipy.optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, res.fun # returns radii, negative max sum
        else:
            # Fallback if LP fails (shouldn't happen with valid centers)
            return np.zeros(n), 0.0
    except Exception:
        return np.zeros(n), 0.0

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    
    # --- Initialization: Hexagonal Grid ---
    # We want to fit 26 circles.
    # Approximate grid size.
    # Try to fit rows.
    # If we use 6 rows, approx 4-5 circles per row.
    
    row_y = []
    circles_per_row = []
    
    # Let's try to fit a hexagonal pattern
    # Width 1, Height 1.
    # Row spacing dy = sqrt(3)/2 * 2r ? No, we don't know r yet.
    # Let's just place points roughly evenly.
    
    # 5 rows of roughly 5 circles?
    # 5, 4, 5, 4, 5, 3 -> sum 26?
    # Let's use a simple uniform grid first for robustness, then relax.
    # 6 columns, 5 rows -> 30 points. Remove 4?
    # Or just place them in a way that covers the square.
    
    # Strategy: 6 rows, 4-5 circles
    rows = 6
    cols = 5 # max
    # 6 * 4 = 24. Need 2 more.
    # Rows 0, 1, 2, 3, 4, 5
    # Lengths: 5, 4, 5, 4, 5, 3? Sum = 26.
    
    # Let's construct centers
    count = 0
    for r in range(rows):
        # Alternate start offset for hexagonal packing
        if r % 2 == 0:
            num_c = 5
            offset_x = 0.0
        else:
            num_c = 4 if r < 5 else 3 # last row smaller?
            if r == 5: num_c = 3
            else: num_c = 4
            offset_x = 1.0 / (num_c + 1) # Shift to center?
            # Actually, for hex packing, shift by half spacing.
            # Spacing approx 1/5 = 0.2. Shift 0.1.
            
        # Uniform distribution in x
        # For row r, place num_c circles
        # If num_c is 5, positions 0.1, 0.3, 0.5, 0.7, 0.9
        # If num_c is 4, positions 0.125, 0.375, 0.625, 0.875?
        
        # Let's just distribute uniformly
        spacing = 1.0 / (num_c + 1)
        for c in range(num_c):
            if count < n:
                y_pos = (r + 0.5) / rows # Centered in y bands
                # Hex shift: if odd row, shift x by spacing/2?
                # Let's keep it simple: uniform grid first, let optimizer fix geometry.
                x_pos = (c + 1) * spacing
                
                # Apply hex shift if row is odd and we want to mimic hex
                # A simple staggered grid is better.
                # Let's do:
                # y = (r + 0.5) / 6
                # x = (c + 0.5 + (r%2)*0.5) / 5  (approx)
                
                centers[count, 0] = x_pos
                centers[count, 1] = y_pos
                count += 1
            else:
                break
                
    # If we didn't fill 26 (logic above was rough), fill remainder randomly
    if count < n:
        for i in range(count, n):
            centers[i, 0] = np.random.rand()
            centers[i, 1] = np.random.rand()

    # --- Optimization Loop ---
    # We will iterate: Solve LP for radii, then move centers apart if constraints are tight.
    
    current_sum_radii = 0
    radii = np.zeros(n)
    
    # Pre-allocate for speed? Not needed for 26.
    
    # Number of iterations
    # LP is fast. We can do ~500 iterations.
    iterations = 300
    alpha = 1.0 # Step size for center update
    
    for step in range(iterations):
        # 1. Solve LP for radii
        radii, neg_obj = solve_radii_lp(centers)
        current_sum_radii = -neg_obj
        
        # 2. Analyze constraints to find forces
        # Calculate distances
        # dists[i, j]
        # We need pairwise dists again.
        # Vectorized distance calculation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Identify tight constraints
        # A constraint is tight if radii[i] + radii[j] is close to dists[i, j]
        # Or if boundary constraint is tight.
        
        # Forces initialization
        forces = np.zeros((n, 2))
        
        # Repulsion between circles
        # We want to increase dists[i, j] if r[i] + r[j] is large relative to dist.
        # Specifically, if r[i] + r[j] > dists[i, j] - epsilon, we have overlap (not possible if LP valid)
        # But if r[i] + r[j] approx dists[i, j], the circle is limiting the radius.
        # We want to push them apart.
        
        # Force magnitude proportional to how much "pressure" is on the constraint.
        # Actually, in the dual of the LP, the dual variable associated with constraint i,j
        # tells us how much the objective would improve if we increased the distance.
        # But we don't have duals easily.
        # Heuristic: Force ~ (r[i] + r[j]) / dists[i, j]^2 ?
        # Or just push apart if they are close.
        
        # Let's use a simple repulsive force model.
        # Force between i and j: F_ij = max(0, (r[i] + r[j] - dists[i, j])) ?
        # But they shouldn't overlap.
        # Better: F_ij = (r[i] + r[j])^2 / dists[i, j]^2 ?
        # Just push centers apart along the line connecting them.
        
        for i in range(n):
            for j in range(i + 1, n):
                d = dists[i, j]
                if d < 1e-9: continue
                sum_r = radii[i] + radii[j]
                
                # If they are close, push apart.
                # The tighter the constraint (sum_r close to d), the more we want to relax it.
                # But sum_r <= d always.
                # Maybe force proportional to sum_r? Larger circles repel more.
                # F ~ (sum_r)^2 / d^2 ?
                
                # Simple repulsion:
                # vector from j to i
                vec = centers[i] - centers[j]
                force_mag = (radii[i] * radii[j]) / (d*d) # Inverse square law weighted by size?
                # Or just 1/d^2
                
                # Let's try: force = (r_i * r_j) / d^2
                # This encourages large circles to stay apart.
                
                forces[i] += (vec / d) * force_mag
                forces[j] -= (vec / d) * force_mag
                
        # Boundary repulsion
        # Push circles away from boundaries if they are large?
        # If circle is close to wall, it limits its own radius.
        # Actually, the LP handles boundary constraints.
        # To increase radius, we want center to be far from boundary.
        # So attractive force to center (0.5, 0.5)?
        # Or repulsive from walls.
        
        for i in range(n):
            x, y = centers[i]
            # Distance to nearest boundary
            dist_to_wall = min(x, 1-x, y, 1-y)
            # If radius is large, we want to be in center.
            # Force towards center (0.5, 0.5)
            vec_to_center = np.array([0.5, 0.5]) - centers[i]
            # Attractive force to center proportional to radius?
            forces[i] += 0.1 * radii[i] * vec_to_center
            
            # Also repulsive from walls
            # If x is small, push right.
            if x < 0.5: forces[i, 0] += 0.01 * radii[i] / (x + 1e-6)
            else: forces[i, 0] -= 0.01 * radii[i] / (1-x + 1e-6)
            
            if y < 0.5: forces[i, 1] += 0.01 * radii[i] / (y + 1e-6)
            else: forces[i, 1] -= 0.01 * radii[i] / (1-y + 1e-6)

        # Update centers
        # Learning rate decay
        step_size = 0.02 / (1 + step/50)
        centers += step_size * forces
        
        # Clip centers to valid range [epsilon, 1-epsilon]
        # Actually, centers must allow for some radius.
        # But if radius is 0, center can be anywhere.
        # Let's keep centers strictly inside to avoid division by zero or numerical issues.
        centers = np.clip(centers, 1e-4, 1 - 1e-4)

    # Final LP solve
    radii, _ = solve_radii_lp(centers)
    
    # Final validation and cleanup
    # Ensure no overlaps due to numerical errors in center update?
    # The LP guarantees radii are valid for the FINAL centers.
    # So (centers, radii) is valid.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
