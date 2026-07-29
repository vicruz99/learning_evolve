# sol_000104 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 80fa60f2) state=270d4cfa sum of radii=2.615938 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses SLSQP optimization with multiple restarts.
    """
    N = 26
    
    # The variable vector structure is [x0, y0, r0, x1, y1, r1, ..., x25, y25, r25]
    # Total dimensions: 3 * N = 78
    
    def objective(vars):
        # We want to maximize sum(radii), so we minimize -sum(radii)
        # Radii are at indices 2, 5, 8, ...
        return -np.sum(vars[2::3])

    def constraints(vars):
        n = N
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        cons = []
        
        # 1. Boundary Constraints:
        # Circle i must be inside [0,1]x[0,1]
        # x_i >= r_i  =>  x_i - r_i >= 0
        cons.extend(x - r)
        # x_i <= 1 - r_i => 1 - x_i - r_i >= 0
        cons.extend(1.0 - x - r)
        # y_i >= r_i  =>  y_i - r_i >= 0
        cons.extend(y - r)
        # y_i <= 1 - r_i => 1 - y_i - r_i >= 0
        cons.extend(1.0 - y - r)
        
        # 2. Non-overlap Constraints:
        # Distance between centers >= sum of radii
        # sqrt((x_i-x_j)^2 + (y_i-y_j)^2) - (r_i + r_j) >= 0
        # We iterate over all unique pairs (i, j) with i < j
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(x[i] - x[j], y[i] - y[j])
                cons.append(dist - (r[i] + r[j]))
                
        return np.array(cons)

    # Define bounds for each variable
    # x, y must be in [0, 1]
    # r must be in [0, 0.5] (radius cannot exceed half the side length)
    bounds = []
    for i in range(N):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    best_sum = -np.inf
    best_vars = None
    
    # Generate initial guesses to restart the optimizer from different points
    # This helps escape local minima
    initial_guesses = []
    
    # Strategy 1: Grid-based initialization
    # Place circles in a grid pattern. This is a valid configuration for small radii.
    # We use a 5-column grid.
    centers_grid = np.zeros((N, 2))
    for i in range(N):
        col = i % 5
        row = i // 5
        # Spacing of 0.2 in x, 0.15 in y fits well
        centers_grid[i, 0] = 0.1 + col * 0.2
        centers_grid[i, 1] = 0.1 + row * 0.15
    
    # Ensure coordinates are within valid range for radius 0.05
    centers_grid[:, 0] = np.clip(centers_grid[:, 0], 0.1, 0.9)
    centers_grid[:, 1] = np.clip(centers_grid[:, 1], 0.1, 0.9)
    
    x0_grid = np.zeros(3 * N)
    x0_grid[0::3] = centers_grid[:, 0]
    x0_grid[1::3] = centers_grid[:, 1]
    x0_grid[2::3] = 0.05  # Start with small valid radii
    initial_guesses.append(x0_grid)
    
    # Strategy 2: Random initialization
    for _ in range(2):
        x0_rand = np.zeros(3 * N)
        x0_rand[0::3] = np.random.uniform(0.2, 0.8, N)
        x0_rand[1::3] = np.random.uniform(0.2, 0.8, N)
        x0_rand[2::3] = 0.05
        initial_guesses.append(x0_rand)

    # Run optimization for each guess
    for x0 in initial_guesses:
        try:
            # SLSQP is suitable for problems with non-linear equality and inequality constraints
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 3000, 'ftol': 1e-12})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_vars = res.x
        except Exception:
            continue

    # Extract results
    if best_vars is not None:
        n = N
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = best_vars[3*i]
            centers[i, 1] = best_vars[3*i+1]
            radii[i] = best_vars[3*i+2]
        
        # Ensure non-negative radii (numerical safety)
        radii = np.maximum(radii, 0.0)
        
        sum_radii = np.sum(radii)
        return centers, radii, sum_radii
    else:
        # Fallback: Return a valid but likely suboptimal packing
        # 6x5 grid with small radius
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        for i in range(N):
            row = i // 6
            col = i % 6
            centers[i, 0] = 0.15 + col * 0.15
            centers[i, 1] = 0.15 + row * 0.15
            radii[i] = 0.05
        return centers, radii, np.sum(radii)
