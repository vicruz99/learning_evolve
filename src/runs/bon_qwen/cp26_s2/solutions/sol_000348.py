# sol_000348 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 03d022f0) state=d930c691 sum of radii=2.461900 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_max_sum_radii(centers):
    """
    Calculates the maximum sum of radii for a fixed set of centers 
    using Linear Programming.
    
    Args:
        centers: np.array of shape (n, 2)
        
    Returns:
        sum_radii: float
        radii: np.array of shape (n)
    """
    n = centers.shape[0]
    
    # Calculate distances between all pairs of centers
    # dist_matrix[i, j] = distance between center i and center j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    
    # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
    c = -np.ones(n)
    
    # Bounds for radii: 0 <= r_i <= min distance to any boundary
    bounds = []
    for i in range(n):
        x, y = centers[i]
        r_bound = min(x, 1 - x, y, 1 - y)
        # Ensure non-negative bound
        r_bound = max(0.0, r_bound)
        bounds.append((0, r_bound))
        
    # Constraints: r_i + r_j <= dist(i, j)
    # We only need to enforce for i < j to avoid redundancy and self-constraints
    # A_ub * r <= b_ub
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    
    row_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[row_idx, i] = 1
            A_ub[row_idx, j] = 1
            b_ub[row_idx] = dist_matrix[i, j]
            row_idx += 1
            
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        radii = res.x
        return -res.fun, radii
    else:
        # Fallback if LP fails (e.g., infeasible due to numerical issues)
        # Return 0 radii
        return 0.0, np.zeros(n)

def objective(centers_flat):
    """
    Objective function for scipy optimizer.
    Minimizes negative sum of radii.
    """
    centers = centers_flat.reshape(-1, 2)
    # Clip centers to [0, 1] to ensure validity during optimization steps
    centers = np.clip(centers, 0, 1)
    sum_r, _ = get_max_sum_radii(centers)
    return -sum_r

def run_packing():
    n_circles = 26
    
    # --- Step 1: Hexagonal Grid Initialization ---
    centers = np.zeros((n_circles, 2))
    
    # Hexagonal parameters
    # We arrange circles in rows. 
    # Vertical spacing: r * sqrt(3)
    # Horizontal spacing: 2 * r
    # Offset for odd rows: r
    
    # We estimate an initial radius to place them nicely. 
    # A radius of 0.1 fits 5 circles in a row. 
    # With 26 circles, we might have 5 rows (e.g., 6, 5, 6, 5, 4 or similar).
    # Let's try to fit a hexagonal pattern that covers the square.
    
    # Approximate number of rows
    # Height ~ 1, row spacing ~ r*sqrt(3). 
    # If r=0.1, spacing ~ 0.173. 1/0.173 ~ 5.7 rows.
    num_rows = 6 
    
    # We will distribute points in a triangular lattice
    # Lattice vectors: v1 = (2, 0), v2 = (1, sqrt(3)) scaled by r
    # We can just generate a grid and pick 26 points that fit best, 
    # or simply place them in a pattern.
    
    # Let's try a specific pattern for 26 circles:
    # 6 rows with varying counts?
    # Or just a dense rectangular patch of hex lattice.
    
    # Let's generate a grid of potential hex centers and pick the best 26
    # that fit in the square with some margin.
    
    r_est = 0.1
    step_x = 2 * r_est
    step_y = np.sqrt(3) * r_est
    
    candidates = []
    # Scan the square with hex lattice
    y = 0
    row_idx = 0
    while y <= 1:
        x_offset = (row_idx % 2) * r_est
        x = x_offset
        while x <= 1:
            # Check if center allows for radius r_est inside square
            if x >= r_est and x <= 1 - r_est and y >= r_est and y <= 1 - r_est:
                candidates.append([x, y])
            x += step_x
        y += step_y
        row_idx += 1
        
    candidates = np.array(candidates)
    
    # If we have more than 26, we need to select 26.
    # A greedy approach: pick points that are far apart?
    # Or just take the first 26 if we ordered them well.
    # Better: take a subset that covers the area well.
    # Actually, just taking a dense subset from the center-out might be good.
    # But for optimization, any valid starting point works.
    # Let's just take the first 26 from our scan if available.
    # If fewer, we generate random points.
    
    if len(candidates) >= n_circles:
        # Select 26 points. 
        # To ensure good distribution, we can pick every k-th point or just slice.
        # Since the scan is dense, slicing might cluster.
        # Let's use a simple sampling.
        indices = np.linspace(0, len(candidates)-1, n_circles, dtype=int)
        centers[:n_circles] = candidates[indices]
    else:
        # Fallback: Random points with some padding
        centers = np.random.uniform(0.1, 0.9, size=(n_circles, 2))

    # --- Step 2: Optimization ---
    # Flatten centers for scipy
    x0 = centers.flatten()
    
    # Bounds for centers: [0, 1]
    bnds = [(0, 1)] * (2 * n_circles)
    
    # Run optimization
    # SLSQP is good for constrained non-linear problems
    # We pass the objective which internally solves LP
    
    result = minimize(objective, x0, method='SLSQP', bounds=bnds, 
                      options={'maxiter': 100, 'ftol': 1e-9})
    
    # Extract final centers
    final_centers = result.x.reshape(-1, 2)
    final_centers = np.clip(final_centers, 0, 1) # Ensure valid
    
    # Calculate final radii using the LP solver to ensure consistency
    sum_r, radii = get_max_sum_radii(final_centers)
    
    return final_centers, radii, sum_r
