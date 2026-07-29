# sol_000247 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 01430d11) state=33736f71 sum of radii=2.448939 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_pairwise_distances(centers):
    """
    Computes the pairwise distances between centers.
    Returns a 1D array of distances for pairs (i, j) with i < j.
    """
    n = centers.shape[0]
    dists = []
    # Vectorized calculation for better performance
    # centers shape: (n, 2)
    # diff shape: (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
    
    # Extract upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dists = dist_matrix[mask]
    return dists

def solve_radii_lp(centers):
    """
    Given centers, solves the LP to maximize sum of radii.
    Returns (sum_radii, radii_array).
    """
    n = centers.shape[0]
    
    # Compute pairwise distances
    dists = get_pairwise_distances(centers)
    
    # LP Variables: r_0, ..., r_{n-1}
    # Objective: Maximize sum(r_i) <=> Minimize -sum(r_i)
    c = -np.ones(n)
    
    # Constraints
    # 1. r_i + r_j <= d_ij for all i < j
    # 2. r_i <= x_i
    # 3. r_i <= 1 - x_i
    # 4. r_i <= y_i
    # 5. r_i <= 1 - y_i
    # 6. r_i >= 0 (handled by bounds)
    
    num_pairs = n * (n - 1) // 2
    num_boundary_constraints = 4 * n
    num_constraints = num_pairs + num_boundary_constraints
    
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    
    # Pairwise constraints
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[idx]
            idx += 1
            
    # Boundary constraints
    for i in range(n):
        # r_i <= x_i
        row = num_pairs + 4 * i
        A_ub[row, i] = 1.0
        b_ub[row] = centers[i, 0]
        
        # r_i <= 1 - x_i
        row += 1
        A_ub[row, i] = 1.0
        b_ub[row] = 1.0 - centers[i, 0]
        
        # r_i <= y_i
        row += 1
        A_ub[row, i] = 1.0
        b_ub[row] = centers[i, 1]
        
        # r_i <= 1 - y_i
        row += 1
        A_ub[row, i] = 1.0
        b_ub[row] = 1.0 - centers[i, 1]
        
    # Bounds: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # Using 'highs' method if available, otherwise default
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        
    if res.success:
        return -res.fun, res.x
    else:
        # If LP fails (should not happen as 0 radii is always feasible), return 0
        return 0.0, np.zeros(n)

def objective(centers_flat):
    """
    Objective function for the optimizer.
    Minimizes negative sum of radii.
    """
    centers = centers_flat.reshape(-1, 2)
    
    # Ensure centers are within [0, 1] to avoid numerical issues in LP bounds
    # Although bounds in minimize should handle this, clipping here is safe.
    centers = np.clip(centers, 0, 1)
    
    val, _ = solve_radii_lp(centers)
    return -val # We want to maximize sum, so minimize negative sum

def initialize_centers(n):
    """
    Initializes centers for n circles.
    Uses a grid pattern with some randomness to break symmetry.
    """
    # Try to fit n circles in a roughly square grid
    # sqrt(26) ~ 5.1
    # 5x6 grid = 30 points. We can pick 26 or just use a dense packing.
    # Let's generate a grid and pick points, or just generate random points
    # and run a quick repulsion pre-step.
    
    # Random initialization within [0.1, 0.9] to keep away from boundaries initially
    # This helps the LP have non-trivial constraints.
    np.random.seed(42) # For reproducibility
    centers = np.random.uniform(0.05, 0.95, size=(n, 2))
    
    # Or a structured initialization
    # Let's try a grid
    grid_size = int(np.ceil(np.sqrt(n)))
    # Create grid points
    xs = np.linspace(0.1, 0.9, grid_size)
    ys = np.linspace(0.1, 0.9, grid_size)
    grid_points = []
    for x in xs:
        for y in ys:
            grid_points.append([x, y])
    
    # If grid has more points than n, shuffle and pick n
    if len(grid_points) > n:
        np.random.shuffle(grid_points)
        centers = np.array(grid_points[:n])
    else:
        centers = np.array(grid_points)
        # If fewer, add random
        while centers.shape[0] < n:
            centers = np.vstack([centers, np.random.uniform(0.1, 0.9, size=(1, 2))])
            
    # Shuffle to avoid ordered structure which might bias optimizer
    np.random.shuffle(centers)
    return centers

def run_packing() -> tuple:
    n = 26
    
    # Initialize centers
    centers_init = initialize_centers(n)
    x0 = centers_init.flatten()
    
    # Define bounds for centers: [0, 1] for each coordinate
    bounds = [(0, 1) for _ in range(2 * n)]
    
    # Optimization
    # Nelder-Mead is a good choice for non-smooth objective functions
    # Maxiter might need to be high, but we have a time limit.
    # Let's try Powell as well, often better for multidimensional.
    
    # We can try multiple restarts to find global optimum
    best_sum = -1e9
    best_centers_flat = x0
    
    # Run 1: From grid initialization
    res1 = minimize(objective, x0, method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    if -res1.fun > best_sum:
        best_sum = -res1.fun
        best_centers_flat = res1.x
        
    # Run 2: From random initialization (different seed)
    np.random.seed(123)
    centers_rand = np.random.uniform(0.05, 0.95, size=(n, 2))
    x0_rand = centers_rand.flatten()
    res2 = minimize(objective, x0_rand, method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    if -res2.fun > best_sum:
        best_sum = -res2.fun
        best_centers_flat = res2.x
        
    # Run 3: Maybe a hexagonal packing start?
    # Hexagonal packing tends to be dense.
    # Let's create a hexagonal grid.
    hex_centers = []
    r_guess = 0.1 # Diameter 0.2
    # Row 0
    y = r_guess
    while y <= 1 - r_guess:
        x = r_guess
        # In hex packing, alternating rows are shifted
        row_idx = int((y - r_guess) / (r_guess * np.sqrt(3)))
        shift = 0
        if row_idx % 2 == 1:
            shift = r_guess # shift by diameter/2? No, diameter/2 is r. 
            # In hex packing, horizontal shift is r (half diameter) if vertical spacing is r*sqrt(3)?
            # Wait, standard hex: centers at (0,0), (d/2, d*sqrt(3)/2).
            # d=2r. Shift is r. Vertical step is r*sqrt(3).
            shift = r_guess 
        
        while x <= 1 - r_guess:
            hex_centers.append([x, y])
            x += 2 * r_guess # Add diameter
        y += r_guess * np.sqrt(3)
        
    # Pick n points from this set or extend it
    # The above generates a specific pattern. It might be too rigid.
    # But let's just use it as another start.
    # We might need to scale or adjust if it doesn't fit well or has < n points.
    # With r=0.1, we can fit many.
    # Let's just take the first n points.
    if len(hex_centers) < n:
        # Pad with random
        while len(hex_centers) < n:
            hex_centers.append([np.random.uniform(0,1), np.random.uniform(0,1)])
    else:
        hex_centers = hex_centers[:n]
        
    x0_hex = np.array(hex_centers).flatten()
    res3 = minimize(objective, x0_hex, method='Nelder-Mead', 
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    if -res3.fun > best_sum:
        best_sum = -res3.fun
        best_centers_flat = res3.x
        
    # Recover final centers and radii
    final_centers = best_centers_flat.reshape(-1, 2)
    _, final_radii = solve_radii_lp(final_centers)
    
    # Verify validity (optional but good for debug)
    # The LP guarantees constraints, but floating point might cause slight issues.
    # We can clamp radii slightly if needed, but LP should be tight.
    # Just ensure non-negative.
    final_radii = np.maximum(final_radii, 0)
    
    total_sum = np.sum(final_radii)
    
    return final_centers, final_radii, total_sum
