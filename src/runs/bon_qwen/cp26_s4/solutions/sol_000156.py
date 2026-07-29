# sol_000156 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a1c97a27) state=6c9019b2 sum of radii=2.601711 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses SLSQP optimization with multiple restarts from hexagonal grid initializations.
    """
    np.random.seed(42)
    n = 26
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Number of random restarts to explore different local optima
    n_restarts = 5
    
    for restart in range(n_restarts):
        # --- Initialization ---
        # Generate a hexagonal-like grid of candidate points
        candidates = []
        spacing = 0.12 # Initial spacing, safe for small radii
        
        # y ranges from spacing to 1-spacing
        y = spacing
        row = 0
        while y <= 1.0 - spacing:
            x_start = spacing
            # Offset every other row for hexagonal packing
            if row % 2 == 1:
                x_start += spacing / 2.0 
            
            x = x_start
            while x <= 1.0 - spacing:
                candidates.append([x, y])
                x += spacing
            y += spacing * np.sqrt(3)/2 # Vertical spacing for hexagonal
            row += 1
            
        # Shuffle to get different subsets for each restart
        np.random.shuffle(candidates)
        
        # Select first n candidates
        if len(candidates) >= n:
            init_centers = np.array(candidates[:n])
        else:
            # Fallback if grid didn't produce enough points (unlikely)
            init_centers = np.array(candidates)
            while len(init_centers) < n:
                init_centers = np.vstack([init_centers, [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]])
        
        # Determine a safe initial radius
        # Must be small enough so circles don't overlap initially
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(init_centers[i] - init_centers[j])
                if d < min_dist:
                    min_dist = d
        
        # Check distance to walls
        dist_to_wall = np.min([np.min(init_centers[:, 0]), np.min(1 - init_centers[:, 0]),
                               np.min(init_centers[:, 1]), np.min(1 - init_centers[:, 1])])
        
        # Safe radius is a fraction of min distance to wall or neighbor
        init_r = min(min_dist / 3.0, dist_to_wall)
        if init_r < 0.005:
            init_r = 0.005
            
        # --- Optimization Setup ---
        # Variables vector: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(n * 3)
        for i in range(n):
            x0[3*i] = init_centers[i, 0]
            x0[3*i+1] = init_centers[i, 1]
            x0[3*i+2] = init_r
            
        # Objective: Maximize sum of radii => Minimize negative sum
        def objective(vars):
            return -np.sum(vars[2::3])
        
        # Constraints
        constraints = []
        
        # 1. Boundary Constraints
        # For each circle i:
        # x_i >= r_i  => x_i - r_i >= 0
        # x_i <= 1 - r_i => 1 - x_i - r_i >= 0
        # y_i >= r_i  => y_i - r_i >= 0
        # y_i <= 1 - r_i => 1 - y_i - r_i >= 0
        # r_i >= 0
        
        # We can vectorize these or add individually. 
        # Adding individually is clear, but vectorizing is faster.
        # However, SLSQP handles list of dicts well. Let's use a single function for boundaries too.
        
        def boundary_constraints(v):
            # v is 1D array of size 3n
            xs = v[0::3]
            ys = v[1::3]
            rs = v[2::3]
            
            vals = []
            # x >= r
            vals.extend(xs - rs)
            # 1 - x - r >= 0
            vals.extend(1.0 - xs - rs)
            # y >= r
            vals.extend(ys - rs)
            # 1 - y - r >= 0
            vals.extend(1.0 - ys - rs)
            # r >= 0
            vals.extend(rs)
            
            return np.array(vals)

        constraints.append({'type': 'ineq', 'fun': boundary_constraints})
        
        # 2. Non-overlap Constraints
        # dist(i,j)^2 >= (r_i + r_j)^2
        # dist^2 - (r_i+r_j)^2 >= 0
        
        def overlap_constraints(v):
            xs = v[0::3]
            ys = v[1::3]
            rs = v[2::3]
            
            # Compute pairwise squared distances
            # Broadcasting: xs[:, None] - xs[None, :]
            dx = xs[:, None] - xs[None, :]
            dy = ys[:, None] - ys[None, :]
            dist_sq = dx**2 + dy**2
            
            # Compute squared sum of radii
            sum_r = rs[:, None] + rs[None, :]
            sum_r_sq = sum_r**2
            
            # We need constraints for i < j
            # Upper triangular indices excluding diagonal
            rows, cols = np.triu_indices(n, k=1)
            
            # Constraint values: dist_sq - sum_r_sq
            # Ensure diagonal is not included (triu_indices with k=1 does this)
            return dist_sq[rows, cols] - sum_r_sq[rows, cols]

        constraints.append({'type': 'ineq', 'fun': overlap_constraints})
        
        # Bounds for variables
        # x, y in [0, 1] (redundant with constraints but helps)
        # r in [0, 0.5]
        bounds = [(0.0, 1.0) for _ in range(n*3)]
        for i in range(n):
            bounds[3*i+2] = (0.0, 0.5)
            
        # --- Run Optimizer ---
        try:
            # SLSQP is suitable for this type of problem
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            
            if res.success:
                current_sum = np.sum(res.x[2::3])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
                    best_radii = np.array([res.x[3*i+2] for i in range(n)])
            else:
                # Even if not successful, might have improved
                current_sum = np.sum(res.x[2::3])
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = np.array([[res.x[3*i], res.x[3*i+1]] for i in range(n)])
                    best_radii = np.array([res.x[3*i+2] for i in range(n)])
                    
        except Exception:
            pass

    # --- Final Result Preparation ---
    if best_centers is None:
        # Fallback to simple grid if everything failed
        best_centers = np.random.uniform(0.2, 0.8, (26, 2))
        best_radii = np.full(26, 0.01)
        best_sum = 0.26
        
    # Ensure no NaNs
    if np.isnan(best_centers).any() or np.isnan(best_radii).any():
        best_centers = np.random.uniform(0.2, 0.8, (26, 2))
        best_radii = np.full(26, 0.01)
        best_sum = 0.26

    return best_centers, best_radii, float(best_sum)
