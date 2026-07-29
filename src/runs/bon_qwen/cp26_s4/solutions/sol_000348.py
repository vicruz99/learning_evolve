# sol_000348 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9cbd6fd8) state=be17de61 sum of radii=2.020663 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_for_radii(centers):
    """
    Solves the Linear Programming problem to find maximum radii 
    for fixed centers.
    """
    n = centers.shape[0]
    if n == 0:
        return 0.0, np.array([])

    # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
    # Variables: r_0, ..., r_{n-1}
    c_obj = np.ones(n) * -1.0
    
    # Constraints matrix A_ub and vector b_ub
    # Constraints:
    # 1. Pairwise non-overlap: r_i + r_j <= dist(c_i, c_j)
    # 2. Boundary: r_i <= dist(c_i, wall)
    
    # Number of constraints
    num_pairs = n * (n - 1) // 2
    num_boundary = 4 * n
    total_constraints = num_pairs + num_boundary
    
    A = np.zeros((total_constraints, n))
    b = np.zeros(total_constraints)
    
    row = 0
    
    # Pairwise constraints
    # Precompute distances to avoid recomputing in loop if needed, 
    # but direct computation is fine here.
    for i in range(n):
        for j in range(i + 1, n):
            # Distance between center i and j
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.sqrt(dx*dx + dy*dy)
            
            # Constraint: 1*r_i + 1*r_j <= d
            A[row, i] = 1.0
            A[row, j] = 1.0
            b[row] = d
            row += 1
            
    # Boundary constraints
    # For each circle i:
    # r_i <= x_i
    # r_i <= 1 - x_i
    # r_i <= y_i
    # r_i <= 1 - y_i
    for i in range(n):
        x, y = centers[i]
        limits = [x, 1.0 - x, y, 1.0 - y]
        for lim in limits:
            A[row, i] = 1.0
            b[row] = lim
            row += 1
            
    # Variable bounds: r_i >= 0
    bounds_r = [(0.0, None)] * n
    
    # Solve LP
    # method='highs' is efficient and robust
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds_r, method='highs')
    
    if res.success:
        max_sum_radii = -res.fun
        radii = res.x
        return max_sum_radii, radii
    else:
        # Fallback if LP fails (should not happen as 0 is feasible)
        return 0.0, np.zeros(n)

def objective_function(centers_flat):
    """
    Objective function for the optimizer.
    Minimizes the negative sum of radii.
    """
    n = 26
    centers = centers_flat.reshape((n, 2))
    val, _ = solve_lp_for_radii(centers)
    return -val

def run_packing():
    n = 26
    
    # --- Initialization ---
    # We want to spread circles out to allow larger radii.
    # A hexagonal packing pattern is a good heuristic.
    # Or a grid. Let's try a perturbed grid.
    # 26 circles. Sqrt(26) approx 5.1. 
    # A 6x5 grid (30 points) is close.
    
    centers = np.zeros((n, 2))
    
    # Create a grid of points
    # We can just place them in a 6x5 grid and take first 26
    # Grid spacing
    cols = 6
    rows = 5
    x_step = 1.0 / (cols + 1)
    y_step = 1.0 / (rows + 1)
    
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx < n:
                # Center the grid somewhat
                # x in (0, 1), y in (0, 1)
                x = (c + 1) * x_step
                y = (r + 1) * y_step
                centers[idx] = [x, y]
                idx += 1
                
    # Add some random perturbation to break symmetry and avoid grid locking
    # Small perturbation is better to stay in valid region
    noise = np.random.normal(0, 0.02, centers.shape)
    centers += noise
    centers = np.clip(centers, 0.01, 0.99) # Keep away from immediate edges
    
    # --- Optimization ---
    # Flatten centers for optimizer
    x0 = centers.flatten()
    
    # Bounds for coordinates [0, 1]
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Use Powell method which is derivative-free and handles bounds well
    # It's good for non-smooth objective functions like this LP value function
    result = minimize(
        objective_function, 
        x0, 
        method='Powell', 
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-9}
    )
    
    best_centers = result.x.reshape((n, 2))
    _, best_radii = solve_lp_for_radii(best_centers)
    sum_radii = np.sum(best_radii)
    
    # Ensure non-negative radii (LP should guarantee, but for safety)
    best_radii = np.maximum(best_radii, 0.0)
    
    return best_centers, best_radii, sum_radii

# Helper to ensure no closures or lambdas are used as per rules
# The functions above are top-level.
