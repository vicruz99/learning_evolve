# sol_000061 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fcf75c21) state=3b6ec6ea sum of radii=1.737449 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import itertools

def solve_radii_for_centers(centers):
    """
    Given fixed centers, solve the LP to maximize sum of radii.
    Variables: r_0, ..., r_25
    Maximize: sum(r_i)
    Subject to:
      r_i >= 0
      r_i <= dist(centroid, boundary) for all 4 sides
      r_i + r_j <= dist(center_i, center_j) for all pairs
    """
    n = centers.shape[0]
    
    # Objective: maximize sum(r) -> minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Constraints matrix A_ub * r <= b_ub
    # We will build this list of constraints
    A_ub = []
    b_ub = []
    
    # Boundary constraints:
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    # These are upper bounds on variables, can be handled by 'bounds' parameter in linprog
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        # Radii must be non-negative
        bounds.append((0, max_r))
        
    # Pairwise constraints: r_i + r_j <= dist_ij
    # Only add if dist_ij is not too large? 
    # Actually, if dist_ij is very large, constraint is loose.
    # But for correctness, we add all. 
    # Optimization: if dist_ij > 1.0 (max possible sum of radii is 1.0 since r<=0.5), 
    # constraint is redundant. But let's just add all for safety, n=26 is small.
    
    # Precompute distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dists[i, j] = np.linalg.norm(centers[i] - centers[j])
            
    # Build sparse or dense matrix? Dense is fine for 26 vars.
    # Number of constraints = n*(n-1)/2 approx 325
    num_pairs = 0
    for i in range(n):
        num_pairs += (n - 1 - i)
        
    A_ub = np.zeros((num_pairs, n))
    b_ub = np.zeros(num_pairs)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            # Constraint: r_i + r_j <= d
            # Row in A_ub: 1 at col i, 1 at col j
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    # Solve LP
    # Method 'highs' is default and efficient
    try:
        res = scipy.optimize.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
        else:
            # If LP fails, return small radii
            return np.zeros(n), 0.0
    except Exception:
        return np.zeros(n), 0.0

def objective_function(centers_flat):
    """
    Wrapper for the optimizer.
    Minimizes negative sum of radii.
    """
    centers = centers_flat.reshape(-1, 2)
    # Clip centers to valid range to prevent weirdness, though bounds handle it
    centers = np.clip(centers, 1e-9, 1 - 1e-9)
    radii, sum_r = solve_radii_for_centers(centers)
    return -sum_r

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Helper to run optimization from a start point
    def optimize_from(centers_init):
        # Flatten centers for scipy
        x0 = centers_init.flatten()
        
        # Bounds for centers: [0, 1]
        bounds_x = [(0, 1)] * (2 * n)
        
        # Use Nelder-Mead as it doesn't require gradients and handles non-smooth landscapes
        # Though it can be slow in high dimensions, 52D is manageable with good init.
        # We use a hybrid approach: maybe a few iterations of Nelder-Mead
        try:
            res = scipy.optimize.minimize(
                objective_function, 
                x0, 
                method='Nelder-Mead', 
                bounds=bounds_x,
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-8}
            )
            if not np.isnan(res.fun):
                return res.x, -res.fun
        except Exception:
            pass
        return x0, objective_function(x0) * -1

    # 1. Initialize with a Hexagonal-like Grid
    # We want to pack 26 circles.
    # Approximate radius for equal circles ~ 0.1. Diameter 0.2.
    # 5 rows of roughly 5 circles.
    # Hexagonal spacing: dx = d, dy = d * sqrt(3)/2
    
    # Let's generate a set of initial centers
    # Strategy: Place points on a grid, then perturb slightly
    centers_list = []
    
    # Try a few different initial configurations
    configs = []
    
    # Config 1: Grid
    # 5x5 grid is 25 points. We need 26.
    # Let's place 26 points in a 6x5 grid area?
    # Just fill space.
    cols = 6
    rows = 5
    xs = np.linspace(0.1, 0.9, cols) # 6 points
    ys = np.linspace(0.1, 0.9, rows) # 5 points
    # Grid has 30 points. We take first 26?
    # Better: distribute evenly.
    # Just use a dense random sample or a specific pattern.
    
    # Let's try a hexagonal packing generation
    # Row 0: 5 circles
    # Row 1: 6 circles (shifted)
    # Row 2: 5 circles
    # Row 3: 6 circles (shifted)
    # Row 4: 4 circles
    # Total: 5+6+5+6+4 = 26
    
    # Estimated radius 0.1.
    # x coords for 5 circles: 0.1, 0.3, 0.5, 0.7, 0.9
    # x coords for 6 circles: 0.03, 0.23, ..., 0.83? 
    # Let's generate coordinates based on a target density.
    
    # Heuristic: Place centers at ( (i + 0.5)/k, (j + 0.5)/m )
    # For 26, maybe 5x6 grid?
    # Let's just use a structured grid of 5x6 and pick 26?
    # Or just 5 rows, varying count.
    
    # Let's create a list of points for a "good" initial guess
    # Hexagonal packing coordinates
    # Let's assume we can fit circles of radius 0.1
    # Spacing 0.2
    points = []
    # Row 0 (y=0.1): 5 points
    for i in range(5):
        points.append([0.1 + i*0.2, 0.1])
    # Row 1 (y=0.1 + 0.1732 = 0.2732): 6 points? 
    # Shift x by 0.1. Start at 0.0? No, must be >= r=0.1.
    # If shift 0.1, x = 0.2, 0.4, 0.6, 0.8. (4 points).
    # To fit 6 points with spacing 0.2, need width 1.0. 
    # 0.1 to 0.9 is width 0.8. Can fit 5 points (0.1, 0.3, 0.5, 0.7, 0.9).
    # Shifted row starts at 0.2, ends at 0.8. 4 points.
    # This suggests hex packing with r=0.1 is tight.
    
    # Let's just use a random perturbation of a grid.
    # Grid 6x5 = 30 points.
    # Take 26 random points from a 6x5 grid?
    # Or just place 26 points in a quasi-random way.
    
    # Sobol sequence or Halton would be good, but simple grid + noise is robust.
    # Let's create a 5x6 grid (30 points), shuffle, take 26.
    grid_xs = np.linspace(0.15, 0.85, 6)
    grid_ys = np.linspace(0.15, 0.85, 5)
    grid_points = []
    for y in grid_ys:
        for x in grid_xs:
            grid_points.append([x, y])
    # 30 points.
    # We need 26.
    # Randomly remove 4?
    np.random.seed(42)
    indices = np.random.choice(30, 26, replace=False)
    init_centers = np.array(grid_points)[indices]
    configs.append(init_centers)
    
    # Config 2: Random points
    np.random.seed(123)
    rand_centers = np.random.uniform(0.05, 0.95, (26, 2))
    configs.append(rand_centers)
    
    # Config 3: Hexagonal pattern specifically tuned
    # 5 rows.
    # y positions: 0.1, 0.28, 0.46, 0.64, 0.82 (approx)
    # x positions:
    # Row 0 (5 pts): 0.1, 0.3, 0.5, 0.7, 0.9
    # Row 1 (6 pts): 0.05, 0.25, 0.45, 0.65, 0.85, 1.05? No.
    # If we allow slightly larger r?
    # Let's just place them.
    hex_centers = []
    # Row 0: 5 pts
    for i in range(5):
        hex_centers.append([0.1 + i*0.2, 0.1])
    # Row 1: 6 pts? Shifted. 
    # To fit 6 pts in [0,1], spacing must be <= 1/5 = 0.2.
    # If spacing 0.2, range is 1.0.
    # 0.0 to 1.0? But boundary constraint r.
    # If r is small, centers can be close to boundary.
    # Let's place them at 0.083, 0.266...
    # Just use a dense hex grid.
    # Let's try to fit 6 points in width 1.
    # x = 1/12, 3/12, ... 11/12?
    # x = 0.0833, 0.25, 0.416, 0.583, 0.75, 0.916.
    # All in [0,1].
    for i in range(6):
        x = (2*i + 1) / 12.0 # 1/12, 3/12...
        # y shift. Height of hex is sqrt(3)/2 * width.
        # width is spacing.
        # Let's assume spacing 1/6 approx 0.166.
        # dy = 0.166 * 0.866 = 0.144.
        y = 0.1 + 0.15 
        hex_centers.append([x, y])
        
    # Row 2: 5 pts (aligned with row 0)
    for i in range(5):
        x = (2*i + 1) / 10.0 # 0.1, 0.3, 0.5, 0.7, 0.9
        y = 0.1 + 2*0.15
        hex_centers.append([x, y])
        
    # Row 3: 6 pts
    for i in range(6):
        x = (2*i + 1) / 12.0
        y = 0.1 + 3*0.15
        hex_centers.append([x, y])
        
    # Row 4: 4 pts? We have 5+6+5+6 = 22. Need 4 more.
    for i in range(4):
        x = (2*i + 2) / 10.0 # 0.2, 0.4, 0.6, 0.8
        y = 0.1 + 4*0.15
        hex_centers.append([x, y])
        
    # Total 26.
    # Check bounds.
    # x range [0.083, 0.916] ok.
    # y range [0.1, 0.1 + 0.6] = 0.7. 
    # Wait, 4*0.15 = 0.6. y_max = 0.7.
    # We can scale y to use full height.
    # Scale y to fit in [0.05, 0.95].
    current_y_min = 0.1
    current_y_max = 0.7
    # Map [current_y_min, current_y_max] to [0.05, 0.95]
    # Actually, just normalize y.
    ys = np.array([p[1] for p in hex_centers])
    ys = (ys - ys.min()) / (ys.max() - ys.min()) * 0.8 + 0.1
    for i in range(len(hex_centers)):
        hex_centers[i][1] = ys[i]
        
    configs.append(np.array(hex_centers))

    # Optimization Loop
    for i, init_c in enumerate(configs):
        # Perturb slightly to break symmetry
        perturbed = init_c + np.random.uniform(-0.02, 0.02, init_c.shape)
        perturbed = np.clip(perturbed, 1e-6, 1-1e-6)
        
        # Run optimizer
        try:
            # We can use 'SLSQP' or 'Powell' or 'Nelder-Mead'.
            # Nelder-Mead is good for non-smooth.
            # But SLSQP might be faster if smooth.
            # Let's try Powell (direction set) which is good for multivariate.
            
            # To save time, limit iterations.
            res = scipy.optimize.minimize(
                objective_function,
                perturbed.flatten(),
                method='Powell',
                bounds=[(0, 1)]*(2*n),
                options={'maxiter': 200, 'ftol': 1e-10}
            )
            
            if not np.isnan(res.fun):
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = res.x.reshape(n, 2)
                    # Recompute radii for final result
                    best_radii, _ = solve_radii_for_centers(best_centers)
        except Exception:
            pass

    # Final validation and adjustment
    if best_centers is not None:
        # Ensure centers are strictly inside
        best_centers = np.clip(best_centers, 1e-9, 1-1e-9)
        best_radii, final_sum = solve_radii_for_centers(best_centers)
        return best_centers, best_radii, final_sum
    else:
        # Fallback to random
        centers = np.random.uniform(0.1, 0.9, (n, 2))
        radii, s = solve_radii_for_centers(centers)
        return centers, radii, s
