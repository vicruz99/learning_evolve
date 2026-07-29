# sol_000006 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96e346d6) state=f6d18d42 sum of radii=2.380506 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import scipy.optimize as opt

def generate_hex_grid(n):
    """
    Generates an initial configuration of n points in a hexagonal lattice pattern
    inside the unit square.
    """
    centers = []
    r_guess = 0.1 # Initial guess radius to guide placement
    
    # Estimate number of rows needed
    # Area ~ n * pi * r^2. But for lattice, density is ~0.9.
    # Just place rows.
    # Vertical spacing sqrt(3)/2 * 2r = sqrt(3)r.
    
    # Let's try to fit rows.
    # Max y for row k (0-indexed) with radius r: r + k * sqrt(3)*r
    # We don't know r yet, but we can place points roughly.
    
    # Heuristic: 5 or 6 rows.
    # Try to distribute n points into rows with sizes alternating.
    rows = []
    cols = int(math.sqrt(n)) + 1
    while sum(len(r) for r in rows) < n:
        # Alternate row lengths to approximate hex packing
        if len(rows) == 0:
            rows.append(list(range(cols)))
        else:
            last_len = len(rows[-1])
            # Shift rows
            new_len = last_len if len(rows) % 2 == 1 else last_len - 1
            if new_len < 1: new_len = 1
            rows.append(list(range(new_len)))
        
    # Flatten and trim/pad
    points = []
    for i, row in enumerate(rows):
        for j in row:
            points.append((j, i))
            
    points = points[:n] # Take first n
    
    # Now assign coordinates.
    # We need to scale these integer coordinates to fit in [0,1]x[0,1]
    # and respect hex geometry.
    
    # Let's determine bounds of the integer grid
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)
    
    # We want to map (j, i) to (x, y)
    # x coordinate: j * spacing_x + offset
    # y coordinate: i * spacing_y
    
    # For hex packing:
    # Horizontal spacing between adjacent circles in a row: 2r
    # Horizontal shift between rows: r (or 0) -> actually centers shift by r if touching.
    # Wait, standard hex: (x, y) -> (x + r, y + sqrt(3)r)
    # So x spacing is 2r? No, distance is 2r.
    # If row 0 is at y=r, x=r, 3r, 5r...
    # Row 1 is at y=r+sqrt(3)r, x=2r, 4r...
    # So horizontal step is 2r. Row shift is r.
    
    # Let's estimate r based on n.
    # Approx area fraction 0.85?
    # n * pi * r^2 approx 0.85 -> r approx sqrt(0.85 / (n*pi))
    r_est = math.sqrt(0.85 / (n * math.pi))
    
    # However, we don't know exact r. Let's just place them in a grid scaled to unit square
    # and then the optimizer will spread them.
    
    # Better initialization:
    # Use a hex grid with spacing 1.0 initially, then scale.
    
    # Let's construct points based on the row structure derived above.
    # Row i has length L_i.
    # Center of row i: y_i = i * math.sqrt(3)/2 * 2 + 1 ?
    # Let's use a continuous parameter for spacing.
    
    # We will just place them with a fixed spacing and let the optimizer handle it.
    # Spacing 0.5 seems safe.
    spacing = 0.2 
    
    final_centers = []
    # Re-generate points properly
    # We need to decide row lengths summing to n.
    # A good pattern for hex packing is alternating lengths.
    # e.g. 5, 4, 5, 4, 5, 3?
    # For n=26.
    # 5 rows of 5? 25.
    # 6 rows? 5,4,5,4,5,4 = 27.
    # Let's target 26.
    # 5, 5, 5, 5, 6? No, hex requires shift.
    # 5, 4, 5, 4, 5, 3 (sum 26).
    
    target_rows = [5, 4, 5, 4, 5, 3]
    if sum(target_rows) > n:
        # trim
        diff = sum(target_rows) - n
        target_rows[-1] -= diff
    elif sum(target_rows) < n:
        diff = n - sum(target_rows)
        target_rows[0] += diff
        
    # Construct coordinates
    # Assume a unit spacing for now (will be scaled)
    # Row 0 (index 0): y=0. x = 0, 2, 4, 6, 8 (for 5 circles)
    # Row 1 (index 1): y=sqrt(3). x = 1, 3, 5, 7 (for 4 circles) -- shifted by 1 unit (which is r in hex logic, but here 2r is step, so shift r)
    # Actually, if step is 2, shift is 1.
    
    pts = []
    y_curr = 0
    for r_idx, count in enumerate(target_rows):
        # Shift for odd rows
        shift = 0.5 if r_idx % 2 == 1 else 0.0 # 0.5 units shift (half step)
        # x positions: 0.5, 2.5, 4.5... or 0, 2, 4...
        # Let's use centers at 0, 2, 4... relative to 0.
        # To center them, we might need offset.
        # Just start at 0.
        
        # Wait, if step is 2, points are 0, 2, 4...
        # Shift 1 unit -> 1, 3, 5...
        
        x_start = shift * 2 # If step is 2, shift is 1. 1 = 0.5 * 2.
        # Let's use step 1 for simplicity in generation, scale later.
        # Step 1: points 0, 1, 2...
        # Hex shift: 0.5
        
        step = 1.0
        shift_offset = 0.5 if r_idx % 2 == 1 else 0.0
        
        for k in range(count):
            x = k * step + shift_offset
            y = r_idx * (math.sqrt(3)/2 * step) # vertical spacing sqrt(3)/2 * step
            pts.append((x, y))
            
    pts = pts[:n]
    
    # Scale pts to fit in [0,1]x[0,1] with some margin
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Add margin 0.1 on each side
    width = max_x - min_x
    height = max_y - min_y
    
    scale_x = 0.8 / width if width > 0 else 1
    scale_y = 0.8 / height if height > 0 else 1
    scale = min(scale_x, scale_y)
    
    # Center the scaled points
    cx = 0.5
    cy = 0.5
    
    centers = []
    for x, y in pts:
        # normalize to 0..1 range first?
        # map min_x to 0, max_x to 1?
        # No, just scale relative to center
        nx = (x - (min_x + max_x)/2) * scale + 0.5
        ny = (y - (min_y + max_y)/2) * scale + 0.5
        centers.append([nx, ny])
        
    return np.array(centers)

def run_packing():
    n = 26
    np.random.seed(42)
    
    # 1. Initial Configuration
    centers = generate_hex_grid(n)
    # Add small random perturbation to break symmetry and avoid singularities
    centers += np.random.uniform(-0.001, 0.001, centers.shape)
    # Clip to valid range (loose)
    centers = np.clip(centers, 0.01, 0.99)
    
    radii = np.ones(n) * 0.1
    
    # 2. Optimization Loop
    # We will use a simple iterative improvement.
    # Objective: Maximize sum of radii.
    # We will treat radii as variables, but keep them equal initially to find a good geometry,
    # then relax.
    
    # Phase 1: Equal Radii Optimization
    # Maximize r such that all constraints satisfied.
    # We can formulate this as minimizing -r subject to constraints.
    # But constraints are non-convex.
    # We will use a local search / gradient ascent on r, moving centers to relieve stress.
    
    # Define a function that calculates the max possible equal radius for a given set of centers
    def get_max_equal_radius(c):
        # Constraint 1: Boundary
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        # r <= min(x, 1-x, y, 1-y) for all circles
        boundary_limit = np.min(np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), 
                                           np.minimum(c[:, 1], 1 - c[:, 1])))
        
        # Constraint 2: Pairwise distance >= 2r
        # 2r <= ||c_i - c_j||
        # r <= ||c_i - c_j|| / 2
        # We need min over all pairs
        
        # Compute all pairwise distances
        # Efficiently
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        # Set diagonal to inf
        dists[np.arange(n), np.arange(n)] = np.inf
        min_dist = np.min(dists)
        
        overlap_limit = min_dist / 2.0
        
        return min(boundary_limit, overlap_limit)

    # We want to maximize this function w.r.t centers.
    # This is a max-min problem.
    # We can use scipy.optimize to maximize this.
    # Since it's non-smooth (min of many functions), we might need care.
    # But `min` is continuous.
    
    # Let's try to optimize centers to maximize the bottleneck radius.
    # We'll use a few restarts or just the initial good guess.
    
    best_r = 0.0
    best_centers = centers
    
    # Optimization using Nelder-Mead or similar for the centers
    # Variable: flattened centers (52 dims)
    # Objective: get_max_equal_radius
    
    # To help the optimizer, we can smooth the function or just rely on local gradient (approximated).
    # Nelder-Mead is derivative-free.
    
    for _ in range(3): # 3 runs with slight noise
        current_centers = centers.copy()
        # Add noise
        current_centers += np.random.uniform(-0.02, 0.02, current_centers.shape)
        current_centers = np.clip(current_centers, 0.05, 0.95)
        
        def objective(x):
            c = x.reshape((n, 2))
            # Penalty for invalid bounds
            if np.any(c < 0.0) or np.any(c > 1.0):
                return -1.0
            return -get_max_equal_radius(c) # Minimize negative radius

        # Bounds for centers: [0, 1]
        bounds = [(0.0, 1.0)] * (2 * n)
        
        # Try to optimize
        # Nelder-Mead doesn't use bounds well, but we can clip or use SLSQP with bounds
        # But SLSQP needs gradients.
        # Let's use COBYLA or just a custom loop.
        
        # Custom Hill Climbing with Adaptive Step
        c = current_centers.copy()
        r = get_max_equal_radius(c)
        step = 0.01
        
        for iteration in range(200):
            # Try to perturb one center at a time to increase r
            improved = False
            for i in range(n):
                best_local_r = r
                best_local_c = c[i].copy()
                
                # Random directions
                for _ in range(5):
                    direction = np.random.randn(2) * step
                    candidate_c = c[i] + direction
                    candidate_c = np.clip(candidate_c, 0.0, 1.0)
                    
                    temp_c = c.copy()
                    temp_c[i] = candidate_c
                    new_r = get_max_equal_radius(temp_c)
                    
                    if new_r > best_local_r + 1e-7:
                        best_local_r = new_r
                        best_local_c = candidate_c
                        improved = True
                
                if best_local_r > r + 1e-7:
                    c[i] = best_local_c
                    r = best_local_r
            
            if not improved:
                step *= 0.5
                if step < 1e-6:
                    break
            else:
                step = min(step * 1.1, 0.1) # Increase step slightly if improving
        
        if r > best_r:
            best_r = r
            best_centers = c.copy()

    # Phase 2: Variable Radii Optimization
    # Now we have good centers. We want to maximize sum of radii.
    # Constraints: r_i + r_j <= ||c_i - c_j||, r_i <= dist(c_i, boundary).
    # This is a Linear Programming problem if centers are fixed.
    # Max sum r_i
    # s.t. r_i + r_j <= D_ij
    #      r_i <= B_i
    #      r_i >= 0
    
    # We can solve this LP using scipy.optimize.linprog.
    
    # Variables: r_1, ..., r_n
    # Objective: minimize -sum(r_i) -> c_obj = -1
    c_obj = -np.ones(n)
    
    # Inequality constraints A_ub x <= b_ub
    # r_i + r_j <= D_ij  ->  [0..1..1..0] x <= D_ij
    # r_i <= B_i -> [0..1..0] x <= B_i
    # -r_i <= 0 (bounds handled by bounds argument)
    
    # Let's build the matrix.
    # Number of constraints: n*(n-1)/2 (pairwise) + n (boundary)
    # n=26 -> 325 + 26 = 351 constraints.
    # Matrix size 351 x 26. Sparse? Not needed for this size.
    
    # Compute distances
    c = best_centers
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    
    # Boundary limits
    boundaries = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), 
                            np.minimum(c[:, 1], 1 - c[:, 1]))
    
    A_ub = []
    b_ub = []
    
    # Pairwise constraints
    # Indices
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    # Boundary constraints
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(boundaries[i])
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for r_i: (0, None)
    bounds_r = [(0, None) for _ in range(n)]
    
    # Solve LP
    res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
    
    if res.success:
        optimal_radii = -res.fun # Wait, res.fun is min sum (-r) = -sum(r). So sum(r) = -res.fun?
        # Actually res.x contains r_i.
        radii = res.x
    else:
        # Fallback to equal radii if LP fails
        radii = np.ones(n) * best_r

    # Verify and cleanup
    # Check if any radii are extremely small or invalid
    radii = np.maximum(radii, 1e-9)
    
    # Calculate sum
    sum_radii = np.sum(radii)
    
    # Final validation check (mental)
    # The LP ensures r_i + r_j <= dist.
    # And r_i <= boundary_dist.
    # So constraints are satisfied.
    
    return best_centers, radii, float(sum_radii)

# Helper for validation logic inside the function if needed, but we rely on math correctness.
# Just to be safe, we can run a quick check.
def check_valid(centers, radii):
    n = centers.shape[0]
    for i in range(n):
        if radii[i] < 0: return False
        x, y = centers[i]
        r = radii[i]
        if x < r or x > 1-r or y < r or y > 1-r: return False
        for j in range(i+1, n):
            d = math.hypot(centers[i][0]-centers[j][0], centers[i][1]-centers[j][1])
            if d < radii[i] + radii[j] - 1e-9: return False
    return True

# The run_packing function must be defined.
# I will include the helper inside or keep it top level.
# The prompt asks for helper functions top level.
