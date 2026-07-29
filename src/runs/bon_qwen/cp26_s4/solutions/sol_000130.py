# sol_000130 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cd61366d) state=aa661581 sum of radii=1.740097 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import minimize

def compute_pairwise_distances(centers):
    """Compute pairwise Euclidean distances between centers."""
    # centers shape: (N, 2)
    # diff shape: (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    return dists

def solve_radii_lp(centers):
    """
    Solve the Linear Program to maximize sum of radii given fixed centers.
    Maximize sum(r_i)
    s.t.
      r_i <= x_i
      r_i <= 1 - x_i
      r_i <= y_i
      r_i <= 1 - y_i
      r_i + r_j <= dist(C_i, C_j) for all i < j
      r_i >= 0
    """
    n = centers.shape[0]
    
    # Objective: maximize sum(r) => minimize -sum(r)
    c = -np.ones(n)
    
    # Constraints matrix A_ub and vector b_ub for A_ub @ r <= b_ub
    # We will collect rows for A_ub and entries for b_ub
    rows = []
    bounds = []
    
    # Boundary constraints: 4 per circle
    # r_i <= x_i  => [0..1..0] r <= x_i
    # r_i <= 1-x_i => [0..1..0] r <= 1-x_i
    # r_i <= y_i
    # r_i <= 1-y_i
    
    # Precompute boundary limits
    x = centers[:, 0]
    y = centers[:, 1]
    
    # We can construct the matrix efficiently
    # Identity matrix for single variable constraints
    I = np.eye(n)
    
    # Rows for r_i <= x_i
    rows.append(I)
    rows.append(I)
    rows.append(I)
    rows.append(I)
    
    b_vals = []
    b_vals.append(x)
    b_vals.append(1 - x)
    b_vals.append(y)
    b_vals.append(1 - y)
    
    # Pairwise constraints: r_i + r_j <= dist
    # This creates a dense band in the matrix, but for N=26 it's small enough.
    # Number of pairs = n*(n-1)/2
    n_pairs = n * (n - 1) // 2
    pair_rows = np.zeros((n_pairs, n))
    pair_b = np.zeros(n_pairs)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_rows[idx, i] = 1.0
            pair_rows[idx, j] = 1.0
            pair_b[idx] = np.sqrt((centers[i, 0] - centers[j, 0])**2 + 
                                  (centers[i, 1] - centers[j, 1])**2)
            idx += 1
            
    rows.append(pair_rows)
    b_vals.append(pair_b)
    
    A_ub = np.vstack(rows)
    b_ub = np.concatenate(b_vals)
    
    # Bounds for r_i: r_i >= 0
    r_bounds = [(0, None) for _ in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=r_bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback to small radii if LP fails (shouldn't happen with 0 radii feasible)
            return np.zeros(n)
    except Exception:
        return np.zeros(n)

def objective_function(centers_flat):
    """
    Objective function to minimize: negative sum of radii.
    centers_flat is a 1D array of size 2*N.
    """
    n = 26
    centers = centers_flat.reshape((n, 2))
    
    # Clip centers to valid range [0, 1] to avoid issues, though optimizer should handle bounds
    # But LP handles boundary constraints based on centers.
    # If centers are outside [0,1], LP might behave oddly?
    # Let's ensure centers are treated as is, but LP constraints r <= x etc handle validity.
    # Actually, if x < 0, r <= x implies r < 0, but r >= 0, so r=0.
    # So it's safe.
    
    radii = solve_radii_lp(centers)
    total_radius = np.sum(radii)
    return -total_radius

def get_best_packing():
    n = 26
    
    # Strategy: Try multiple initial configurations and optimize centers
    # Using Nelder-Mead or Powell on the centers
    
    configs = []
    
    # 1. Hexagonal Grid Initialization
    # Try to fit 26 circles in a hex pattern
    # Approx radius 0.1. Spacing 0.2.
    # Rows and columns logic
    row_y = []
    row_x_starts = []
    
    # Let's try a 5x5 grid perturbed or a specific hex layout
    # Hex layout:
    # Row 0: 5 circles
    # Row 1: 5 circles
    # Row 2: 6 circles
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Total 26.
    
    # Coordinates relative to a bounding box, then scale to [0.1, 0.9]
    # Or just place in [0,1] directly.
    
    hex_centers = []
    
    # Let's define a layout
    # 5 rows. 
    # Row 0 (y=0.1): 5 circles at x = 0.1, 0.3, 0.5, 0.7, 0.9
    # Row 1 (y=0.273): 5 circles shifted by 0.1? x = 0.2, 0.4, 0.6, 0.8... wait 0.2 to 0.8 is 4 intervals -> 5 circles.
    # Row 2 (y=0.446): 6 circles? Maybe hard to fit 6.
    # Let's just use a dense random seed or a grid.
    
    # Config 1: Grid 5x6 (30 points) -> take first 26
    grid_x = np.linspace(0.1, 0.9, 6)
    grid_y = np.linspace(0.1, 0.9, 5)
    xs, ys = np.meshgrid(grid_x, grid_y)
    grid_points = np.column_stack((xs.flatten(), ys.flatten()))
    # Take 26 points
    if len(grid_points) >= n:
        configs.append(grid_points[:n])
    
    # Config 2: Random
    rng = np.random.RandomState(42)
    rand_centers = rng.uniform(0.1, 0.9, size=(n, 2))
    configs.append(rand_centers)
    
    # Config 3: Hexagonal Lattice approximation
    # Generate points on hex lattice and filter/scale
    hex_points = []
    # Hex lattice vectors
    # (1, 0) and (0.5, sqrt(3)/2)
    # Scale factor s
    # Try s = 0.2
    s = 0.18
    points = []
    for i in range(10):
        for j in range(10):
            x = i * s + (j % 2) * s * 0.5
            y = j * s * np.sqrt(3) / 2
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                points.append([x, y])
    
    if len(points) >= n:
        # Select n points that are well spread?
        # Just take first n
        configs.append(np.array(points[:n]))

    best_centers = None
    best_val = -np.inf
    
    # Optimization settings
    # Powell is good for derivative-free optimization
    for i, init_centers in enumerate(configs):
        # Reshape to 1D
        x0 = init_centers.flatten()
        
        # Bounds for centers: [0, 1]
        bnds = [(0, 1) for _ in range(2 * n)]
        
        try:
            res = minimize(objective_function, x0, method='Powell', bounds=bnds, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            if res.success or (-res.fun > best_val):
                best_val = -res.fun
                best_centers = res.x.reshape((n, 2))
        except Exception as e:
            print(f"Optimization failed for config {i}: {e}")
            continue

    # If optimization didn't improve or failed, fallback to init
    if best_centers is None:
        best_centers = configs[0]

    # Final LP solve to get exact radii for the best centers
    final_radii = solve_radii_lp(best_centers)
    final_sum = np.sum(final_radii)
    
    return best_centers, final_radii, final_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    centers, radii, sum_radii = get_best_packing()
    return centers, radii, sum_radii
