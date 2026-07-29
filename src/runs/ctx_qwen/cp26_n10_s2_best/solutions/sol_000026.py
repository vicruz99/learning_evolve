# sol_000026 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d17cbe8) state=f081a56f sum of radii=2.602828 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v, n):
    """
    Objective function to minimize.
    We want to maximize the sum of radii, so we minimize the negative sum.
    Variables v are ordered: [x1...xn, y1...yn, r1...rn]
    """
    radii = v[2*n:]
    return -np.sum(radii)

def constraints_fun(v, n):
    """
    Vectorized constraint function.
    Returns an array where all elements must be >= 0.
    """
    centers_x = v[:n]
    centers_y = v[n:2*n]
    radii = v[2*n:]
    
    cons_list = []
    
    # 1. Boundary constraints
    # x - r >= 0
    cons_list.append(centers_x - radii)
    # 1 - x - r >= 0
    cons_list.append(1 - centers_x - radii)
    # y - r >= 0
    cons_list.append(centers_y - radii)
    # 1 - y - r >= 0
    cons_list.append(1 - centers_y - radii)
    
    # 2. Pairwise non-overlap constraints
    # dist^2 >= (r_i + r_j)^2
    # Compute difference matrices for x and y coordinates
    diff_x = centers_x[:, np.newaxis] - centers_x[np.newaxis, :]
    diff_y = centers_y[:, np.newaxis] - centers_y[np.newaxis, :]
    
    # Squared Euclidean distances
    dist_sq = diff_x**2 + diff_y**2
    
    # Squared sum of radii
    sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
    sum_r_sq = sum_r**2
    
    # We only need constraints for i < j (upper triangle)
    # Create a boolean mask for the upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Extract valid constraints
    pair_cons = (dist_sq - sum_r_sq)[mask]
    
    cons_list.append(pair_cons)
    
    return np.concatenate(cons_list)

def generate_initial_guess(n, seed):
    """
    Generates an initial valid configuration based on a hexagonal lattice.
    """
    np.random.seed(seed)
    # Estimated radius for a dense packing. 
    # Hexagonal packing is denser, allowing slightly larger radii than grid.
    r_est = 0.095 
    
    centers = []
    y = r_est
    row = 0
    
    # Generate points in hexagonal pattern until we have enough
    while len(centers) < n + 10:
        # Shift odd rows by r_est
        x_start = r_est + (row % 2) * r_est
        x = x_start
        while x <= 1 - r_est:
            centers.append([x, y])
            x += 2 * r_est
        # Vertical spacing for hexagonal packing is sqrt(3) * r
        y += np.sqrt(3) * r_est
        row += 1
    
    # Select first n points
    selected = np.array(centers[:n])
    
    # Add small random perturbation to break symmetry and help optimization
    selected += np.random.uniform(-0.002, 0.002, size=selected.shape)
    
    # Clip to ensure centers are strictly inside (with margin)
    selected = np.clip(selected, 0.01, 0.99)
    
    return selected

def run_packing():
    n = 26
    
    # Define bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x bounds
    for _ in range(n):
        bounds.append((0, 1)) # y bounds
    for _ in range(n):
        bounds.append((0, 0.5)) # r bounds
        
    # Define constraints
    cons = {'type': 'ineq', 'fun': constraints_fun, 'args': (n,)}
    
    best_result = None
    best_sum = -1.0
    
    # Run optimization with multiple restarts to avoid local minima
    num_restarts = 4
    for i in range(num_restarts):
        # Generate initial guess
        centers_init = generate_initial_guess(n, seed=i*100 + 7)
        
        # Extract initial x, y
        x_init = centers_init[:, 0]
        y_init = centers_init[:, 1]
        
        # Initialize radii to a small value to ensure initial feasibility
        # Distance between centers is ~0.19, sum of radii 0.02 -> valid
        r_init = np.full(n, 0.01)
        
        # Concatenate variables: [x..., y..., r...]
        x0 = np.concatenate([x_init, y_init, r_init])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, args=(n,),
                           constraints=cons, options={'maxiter': 500, 'ftol': 1e-12})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = res
        except Exception:
            # If optimization fails for this restart, try next
            pass
            
    if best_result is not None:
        res = best_result
        x_sol = res.x[:n]
        y_sol = res.x[n:2*n]
        r_sol = res.x[2*n:]
        centers_sol = np.column_stack((x_sol, y_sol))
        
        # Ensure radii are non-negative (optimizer should handle, but for safety)
        r_sol = np.maximum(r_sol, 0.0)
        
        return centers_sol, r_sol, float(best_sum)
    else:
        # Fallback solution: 6x6 grid subset with r=0.08
        # This is valid (dist=0.16, 2r=0.16) but suboptimal (sum=2.08)
        coords = [0.08 + k*0.16 for k in range(6)]
        centers = []
        radii = []
        count = 0
        for x in coords:
            for y in coords:
                if count < n:
                    centers.append([x, y])
                    radii.append(0.08)
                    count += 1
        return np.array(centers), np.array(radii), 26 * 0.08
