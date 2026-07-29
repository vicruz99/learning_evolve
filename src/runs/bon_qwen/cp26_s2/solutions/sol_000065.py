# sol_000065 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4705e2a5) state=e5daa2ce sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    """
    n = 26
    # Initial radii guess (small enough to be valid with any distinct centers)
    initial_r = 0.02
    
    # Initialize centers in a structured hexagonal-like grid to ensure good spacing
    centers = np.zeros((n, 2))
    
    # Hexagonal layout parameters
    row_y = 0.12  # Vertical offset
    row_spacing = 0.18
    base_x = 0.12 # Horizontal offset
    x_spacing = 0.22
    
    idx = 0
    # Generate 5 rows with varying number of circles to approximate 26
    # Row configurations: 6, 5, 6, 5, 4 -> Sum = 26
    row_counts = [6, 5, 6, 5, 4]
    
    for row_idx, count in enumerate(row_counts):
        y = row_y + row_idx * row_spacing
        # Shift every other row for hexagonal packing
        x_offset = base_x if row_idx % 2 == 0 else base_x + x_spacing / 2
        
        for col in range(count):
            if idx < n:
                x = x_offset + col * x_spacing
                # Clamp coordinates to ensure valid start
                x = np.clip(x, 0.05, 0.95)
                y = np.clip(y, 0.05, 0.95)
                centers[idx] = [x, y]
                idx += 1
                
    # Ensure we filled all 26 spots (fallback to random if logic fails)
    while idx < n:
        centers[idx] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
        idx += 1
        
    radii = np.full(n, initial_r)

    # --- Optimization Setup ---

    def objective(z):
        # z shape (3n,) -> [x1, y1, r1, x2, y2, r2, ...]
        # Return negative sum of radii to minimize
        return -np.sum(z[2::3])

    def constraints_func(z):
        # Returns a vector of values >= 0
        X = z[0::3]
        Y = z[1::3]
        R = z[2::3]
        
        constraints = []
        
        # 1. Boundary constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
        # Constraints must be >= 0
        constraints.extend(X - R)        # x - r >= 0
        constraints.extend((1 - X) - R)  # 1 - x - r >= 0
        constraints.extend(Y - R)        # y - r >= 0
        constraints.extend((1 - Y) - R)  # 1 - y - r >= 0
        
        # 2. Non-overlap constraints: dist(i, j) - (r_i + r_j) >= 0
        # We compute the lower triangle of the distance matrix minus radius sum
        # Using broadcasting for efficiency
        X_col = X[:, np.newaxis]
        Y_col = Y[:, np.newaxis]
        R_col = R[:, np.newaxis]
        
        # Squared distances to avoid sqrt in constraint calculation if possible?
        # No, SLSQP handles non-linear smooth functions. Sqrt is fine.
        # But calculating dist matrix is O(N^2). N=26 is small.
        
        # Distance matrix
        dx = X_col - X_col.T
        dy = Y_col - Y_col.T
        dists = np.sqrt(dx**2 + dy**2)
        
        # Radius sum matrix
        r_sums = R_col + R_col.T
        
        # Overlap violation: dists - r_sums. We only need lower triangle (i > j)
        # Use np.tril_indices to get lower triangle elements
        lower_idx = np.tril_indices(n, -1) # -1 to exclude diagonal
        non_overlap = dists[lower_idx] - r_sums[lower_idx]
        
        constraints.extend(non_overlap)
        
        return np.array(constraints)

    # Define constraints for SLSQP
    # We pass the function directly. SLSQP evaluates it.
    cons = {'type': 'ineq', 'fun': constraints_func}
    
    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (loose upper bound)
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Initial guess vector
    z0 = np.zeros(3 * n)
    for i in range(n):
        z0[3*i] = centers[i, 0]
        z0[3*i+1] = centers[i, 1]
        z0[3*i+2] = radii[i]
        
    # Run Optimization
    # SLSQP is suitable for this type of non-linear constrained problem
    result = minimize(objective, z0, method='SLSQP', bounds=bounds, 
                      constraints=cons, 
                      options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False})
    
    # Extract results
    z_opt = result.x
    centers_opt = np.zeros((n, 2))
    radii_opt = np.zeros(n)
    
    for i in range(n):
        centers_opt[i, 0] = z_opt[3*i]
        centers_opt[i, 1] = z_opt[3*i+1]
        radii_opt[i] = z_opt[3*i+2]
        
    # Post-processing: slightly shrink radii to strictly satisfy validation tolerance (1e-12)
    # The optimizer might touch the boundary exactly.
    epsilon = 1e-4
    radii_opt = np.maximum(radii_opt - epsilon, 0.0)
    
    # Recalculate sum of radii
    total_sum = np.sum(radii_opt)
    
    return centers_opt, radii_opt, total_sum
