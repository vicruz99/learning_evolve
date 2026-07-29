# sol_000013 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9f77b693) state=e074368f sum of radii=2.502871 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # Function to solve the packing problem for a given initial guess
    def solve_packing(x0):
        # Bounds for variables: x in [0,1], y in [0,1], r in [0, 0.5]
        # We tighten r upper bound to 0.5 (max possible)
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
        
        # Constraints definition
        # 1. Boundary constraints:
        #    x_i >= r_i  => x_i - r_i >= 0
        #    1 - x_i >= r_i => 1 - x_i - r_i >= 0
        #    Same for y
        # 2. Pairwise non-overlap:
        #    dist_ij^2 >= (r_i + r_j)^2 => dist_ij^2 - (r_i + r_j)^2 >= 0

        def objective(vars):
            # vars layout: [x1, y1, r1, x2, y2, r2, ...]
            # We want to maximize sum(r), so minimize -sum(r)
            r = vars[2::3]
            return -np.sum(r)

        def boundary_constraints(vars):
            # Vectorized extraction
            X = vars[0::3]
            Y = vars[1::3]
            R = vars[2::3]
            
            # Returns array of constraint values >= 0
            # [x1-r1, 1-x1-r1, y1-r1, 1-y1-r1, ...]
            cons = np.zeros(4 * n)
            cons[0::4] = X - R
            cons[1::4] = 1 - X - R
            cons[2::4] = Y - R
            cons[3::4] = 1 - Y - R
            return cons

        def pairwise_constraints(vars):
            # Vectorized extraction
            X = vars[0::3]
            Y = vars[1::3]
            R = vars[2::3]
            
            # Compute distance matrix and radius sum matrix
            # Using broadcasting
            # X shape (n,), X[:, None] shape (n, 1)
            diff_X = X[:, None] - X[None, :] # (n, n)
            diff_Y = Y[:, None] - Y[None, :] # (n, n)
            dist_sq = diff_X**2 + diff_Y**2
            
            sum_R = R[:, None] + R[None, :] # (n, n)
            r_sum_sq = sum_R**2
            
            # Constraint: dist_sq - r_sum_sq >= 0
            # We only need upper triangle (i < j)
            # Flatten and select upper triangle indices
            constraint_vals = dist_sq - r_sum_sq
            
            # Mask for upper triangle (strictly above diagonal)
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            return constraint_vals[mask]

        cons = [
            {'type': 'ineq', 'fun': boundary_constraints},
            {'type': 'ineq', 'fun': pairwise_constraints}
        ]

        # Run optimization
        # Using SLSQP which supports bounds and constraints
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
        
        return res

    best_sum = -np.inf
    best_centers = None
    best_radii = None

    # Run multiple times with different random seeds to avoid local minima
    # 5x5 grid initialization is also a good deterministic start
    # Let's try 3 random initializations + 1 grid initialization
    
    seeds = [42, 123, 456]
    
    # 1. Grid initialization (5x5 + 1)
    # 5x5 grid points
    grid_x = np.linspace(0.1, 0.9, 5)
    grid_y = np.linspace(0.1, 0.9, 5)
    cx, cy = np.meshgrid(grid_x, grid_y)
    centers_grid = np.vstack([cx.ravel(), cy.ravel()]).T # 25 points
    
    # Add 26th point in a gap, e.g., center of square if not present? 
    # 0.5 is in grid. Let's place it slightly offset or random.
    # Or just place it at (0.5, 0.5) is already there.
    # Let's add a point at (0.2, 0.8) ? No, just random near center.
    extra_point = np.array([[0.5, 0.5]]) # Overlap, optimizer will move it
    # Better: add point at (0.5, 0.5) is duplicate. 
    # Let's just use random for 26th.
    rng_init = np.random.RandomState(0)
    extra = rng_init.uniform(0.2, 0.8, 2)
    centers_grid = np.vstack([centers_grid, extra])
    
    # Initial radii: small enough to be valid
    # Estimate based on grid spacing 0.2 -> r=0.1. 
    # But with 26th point, maybe smaller.
    init_r = 0.05 
    
    x0_grid = np.zeros(3 * n)
    x0_grid[0::3] = centers_grid[:, 0]
    x0_grid[1::3] = centers_grid[:, 1]
    x0_grid[2::3] = init_r
    
    res_grid = solve_packing(x0_grid)
    if res_grid.success:
        centers = np.vstack([res_grid.x[0::3], res_grid.x[1::3]]).T
        radii = res_grid.x[2::3]
        s = np.sum(radii)
        if s > best_sum:
            best_sum = s
            best_centers = centers.copy()
            best_radii = radii.copy()

    # 2. Random initializations
    for seed in seeds:
        rng = np.random.RandomState(seed)
        # Random positions in [0.1, 0.9] to stay away from boundaries initially
        x = rng.uniform(0.1, 0.9, n)
        y = rng.uniform(0.1, 0.9, n)
        r = 0.04 * np.ones(n) # Small radii
        
        x0 = np.zeros(3 * n)
        x0[0::3] = x
        x0[1::3] = y
        x0[2::3] = r
        
        res = solve_packing(x0)
        if res.success:
            centers = np.vstack([res.x[0::3], res.x[1::3]]).T
            radii = res.x[2::3]
            s = np.sum(radii)
            if s > best_sum:
                best_sum = s
                best_centers = centers.copy()
                best_radii = radii.copy()

    # If no solution found (unlikely), return zeros or safe default
    if best_centers is None:
        # Fallback
        centers = np.random.rand(n, 2)
        radii = np.zeros(n)
        best_sum = 0.0

    # Final validation/cleanup
    # Ensure radii are non-negative
    best_radii = np.maximum(best_radii, 0)
    
    # Clamp centers to [0, 1]
    best_centers = np.clip(best_centers, 0, 1)
    
    # Re-calculate sum
    final_sum = np.sum(best_radii)
    
    return best_centers, best_radii, final_sum
