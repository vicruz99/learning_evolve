# sol_000186 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1380b4f2) state=103a5688 sum of radii=1.815442 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle

    Returns:
        True if valid, False otherwise
    """
    n = centers.shape[0]

    # Check for NaN values
    if np.isnan(centers).any():
        print("NaN values detected in circle centers")
        return False

    if np.isnan(radii).any():
        print("NaN values detected in circle radii")
        return False

    # Check if radii are nonnegative and not nan
    for i in range(n):
        if radii[i] < 0:
            print(f"Circle {i} has negative radius {radii[i]}")
            return False
        elif np.isnan(radii[i]):
            print(f"Circle {i} has nan radius")
            return False

    # Check if circles are inside the unit square
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            print(f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square")
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:  # Allow for tiny numerical errors
                print(f"Circles {i} and {j} overlap: dist={dist}, r1+r2={radii[i]+radii[j]}")
                return False

    return True

def get_max_equal_radius(centers):
    """
    Given centers, calculate the maximum equal radius r such that 
    circles don't overlap and are inside [0,1]^2.
    """
    n = centers.shape[0]
    min_dist = float('inf')
    
    # Check distance to boundaries
    for i in range(n):
        x, y = centers[i]
        d_boundary = min(x, 1-x, y, 1-y)
        if d_boundary < min_dist:
            min_dist = d_boundary
            
    # Check distance between centers
    # r + r <= dist => 2r <= dist => r <= dist/2
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < min_dist * 2: # Optimization: only check if relevant
                 # We need r <= dist/2
                 val = dist / 2.0
                 if val < min_dist:
                     min_dist = val

    return min_dist

def objective_neg_sum_radii(params, n=26):
    """
    Objective function to maximize sum of radii.
    We optimize centers. Radii are derived from centers.
    To make it smooth for L-BFGS-B, we can use a penalty method or 
    just optimize the minimum radius (equal radii case) first.
    Here we will optimize for equal radii first to get a good baseline,
    then adjust.
    
    Actually, for the optimizer, let's maximize the minimum radius (equal radii).
    Maximize r s.t. constraints.
    Equivalent to minimize -r.
    """
    centers = params.reshape((n, 2))
    r = get_max_equal_radius(centers)
    return -r

def generate_hexagonal_lattice(n=26):
    """
    Generates a hexagonal lattice configuration for n circles in [0,1]x[0,1].
    """
    # Approximate radius for 26 circles in square
    # Area approx: 26 * pi * r^2 <= 1 => r ~ 0.11
    # But boundary effects reduce this. r ~ 0.1 is safe start.
    
    # Try to fit rows
    # Hexagonal packing: rows offset by r horizontally, vertical spacing sqrt(3)/2 * 2r = sqrt(3)r
    
    # Let's try a grid approach that is dense
    # 5x5 grid is 25. 6th is extra.
    # Maybe 5 rows with lengths 5, 5, 6, 5, 5? No, 6 is too wide for r>0.08.
    # Maybe 5, 5, 5, 5, 6?
    
    # Let's use a generic hexagonal generation and clip/scale
    points = []
    # Spacing
    s_x = 1.0 / 5.5 # slightly less than 0.2
    s_y = 1.0 / 5.5
    
    # Try placing in a triangular grid
    # y = k * sqrt(3)/2 * s_x
    # x = j * s_x + (k%2) * s_x/2
    
    # We want to cover the square.
    # Let's just generate many points and pick 26 best?
    # Or just construct 26.
    
    # Heuristic: 5 rows.
    # Row 0: 5 pts
    # Row 1: 5 pts (shifted)
    # Row 2: 6 pts? No.
    # Let's do 5, 5, 5, 5, 6?
    # If r=0.1, width 1.0. 6 circles need 1.2 width. Impossible.
    # So max 5 per row if axis aligned.
    # But hexagonal allows nesting.
    # If row 1 has 5, row 2 has 4 nestled?
    # 5+4+5+4+5+3 = 26.
    
    rows_counts = [5, 4, 5, 4, 5, 3]
    points = []
    
    # Scale factor to fit in 1x1
    # Height: (num_rows - 1) * vertical_spacing + 2*r
    # Width: max(width of rows)
    
    # Let's assume r=0.1 initially to layout, then scale?
    # No, coordinates are absolute.
    
    # Let's try to fit in [0,1]x[0,1] directly.
    # Vertical spacing dy
    # Horizontal spacing dx = 2r?
    
    # Let's use a standard hex grid generator
    # r_est = 0.1
    # dx = 2*r_est
    # dy = math.sqrt(3)*r_est
    
    # We have 6 rows.
    # y_i = r_est + i * dy
    # Last y: r_est + 5*dy = 0.1 + 5*0.1732 = 0.966. Fits.
    
    r_est = 0.1
    dy = math.sqrt(3) * r_est
    
    current_points = []
    row_idx = 0
    for count in rows_counts:
        y = r_est + row_idx * dy
        # Shift x based on row parity for hexagonal
        shift = (row_idx % 2) * (2 * r_est / 2.0) # shift by r
        
        # Distribute 'count' circles in x with spacing 2r
        # Total width needed: (count-1)*2r + 2r = count*2r
        # We need to fit in [0,1].
        # If count=5, width 1.0. Fits exactly.
        # If count=4, width 0.8. Center it.
        # If count=3, width 0.6. Center it.
        
        width_needed = count * 2 * r_est
        start_x = (1.0 - width_needed) / 2.0 + shift
        
        for k in range(count):
            x = start_x + k * (2 * r_est)
            current_points.append([x, y])
        row_idx += 1
        
    return np.array(current_points)

def run_packing():
    n = 26
    
    # 1. Generate initial guess
    centers = generate_hexagonal_lattice(n)
    
    # 2. Optimize centers to maximize minimum radius (Equal Radii Packing)
    # We minimize -r.
    # Bounds for centers [0, 1]
    bounds = [(0.0, 1.0) for _ in range(2 * n)]
    
    # Use L-BFGS-B with multiple restarts or just one good run
    # The function is non-convex, so we might get stuck.
    # Let's run it a few times with random perturbations.
    
    best_params = centers.flatten()
    best_val = -get_max_equal_radius(centers) # minimize -r
    
    # Run optimization
    # To help the optimizer, we can use a smoother objective or just the raw one.
    # The raw min function is non-differentiable at points where the active constraint changes.
    # But L-BFGS-B usually handles it okay or we can use SLSQP.
    
    # Let's try to optimize directly.
    # Using a wrapper to handle the min function.
    
    # To make it robust, we can use differential evolution for global search if time permits,
    # but for 52 vars it's slow.
    # Let's stick to local search from a good start.
    
    # Perturb initial guess slightly
    np.random.seed(42)
    perturbed_centers = centers + np.random.normal(0, 0.01, centers.shape)
    # Clip to bounds
    perturbed_centers = np.clip(perturbed_centers, 0.01, 0.99)
    
    res = scipy.optimize.minimize(
        objective_neg_sum_radii,
        perturbed_centers.flatten(),
        args=(n,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6}
    )
    
    centers_opt = res.x.reshape((n, 2))
    
    # 3. Calculate equal radii
    r_eq = get_max_equal_radius(centers_opt)
    radii_eq = np.full(n, r_eq)
    sum_radii_eq = np.sum(radii_eq)
    
    # 4. Refine with variable radii?
    # For fixed centers, the problem of maximizing sum of radii is an LP.
    # However, we can approximate by setting each r_i to its max possible value
    # given the centers and other radii?
    # Actually, for fixed centers, we can just solve:
    # Maximize sum r_i
    # s.t. r_i <= dist_to_wall_i
    #      r_i + r_j <= dist_ij
    #      r_i >= 0
    # This is a linear program.
    
    # Let's set up the LP
    # Variables: r_0, ..., r_25
    # Constraints:
    # r_i <= wall_dist_i  => r_i <= c_i
    # r_i + r_j <= d_ij   => r_i + r_j <= d_ij
    
    # We can use scipy.optimize.linprog?
    # Minimize -sum(r_i) => c_obj = -1
    # Constraints: A_ub x <= b_ub
    
    wall_dists = np.array([min(c[0], 1-c[0], c[1], 1-c[1]) for c in centers_opt])
    
    # A_ub matrix construction
    # Rows for wall constraints: I
    # Rows for pair constraints: e_i + e_j
    
    # Number of variables: n
    A_ub = []
    b_ub = []
    
    # Wall constraints
    for i in range(n):
        row = [0.0] * n
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(wall_dists[i])
        
    # Pair constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j]) ** 2))
            row = [0.0] * n
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c_obj = -np.ones(n)
    bounds_lp = [(0, None)] * n
    
    from scipy.optimize import linprog
    res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
    
    if res_lp.success:
        radii_var = res_lp.x
        sum_radii_var = np.sum(radii_var)
    else:
        radii_var = radii_eq
        sum_radii_var = sum_radii_eq
        
    # Use the better of equal or variable radii solution
    # Note: The variable radii solution is guaranteed >= equal radii solution for fixed centers.
    
    final_centers = centers_opt
    final_radii = radii_var
    final_sum = sum_radii_var
    
    # Validate
    # The LP ensures non-overlap and boundary constraints with epsilon 0?
    # LP uses exact arithmetic (mostly), but floating point might have issues.
    # The constraints are <= dist.
    # If dist = r_i + r_j, they touch.
    # We need strict inequality or tolerance?
    # The validation allows 1e-12 tolerance.
    # LP might return r_i + r_j = dist exactly.
    # We should shrink radii slightly to be safe.
    
    # Let's shrink by a tiny factor
    shrink_factor = 0.999999
    final_radii = final_radii * shrink_factor
    
    # Re-validate just to be sure (internal check)
    # If validation fails, we might need to reduce radii more.
    # But with 1e-12 tolerance, 0.999999 should be fine if LP was tight.
    
    return final_centers, final_radii, np.sum(final_radii)

# Note: The prompt requires run_packing to be defined and returned.
# The code above defines it.
