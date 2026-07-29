# sol_000251 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2f3e441d) state=cbe58771 sum of radii=2.610767 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # Optimization parameters
    n_restarts = 5
    initial_grid_size = 6
    
    for _ in range(n_restarts):
        # 1. Initialization: Staggered Hexagonal Grid with noise
        centers = []
        for r in range(initial_grid_size):
            for c in range(initial_grid_size):
                x = (c + 0.5 + (0.5 if r % 2 == 1 else 0.0)) / initial_grid_size
                y = (r + 0.5) / initial_grid_size
                
                # Perturbation to help escape local minima
                x += np.random.uniform(-0.05, 0.05)
                y += np.random.uniform(-0.05, 0.05)
                
                centers.append([x, y])
                if len(centers) == n:
                    break
            if len(centers) == n:
                break
        
        centers = np.array(centers)
        radii = np.full(n, 0.09) # Start with a reasonable radius
        
        # Flatten variables for scipy
        x0 = np.concatenate([centers.flatten(), radii])
        
        # 2. Objective Function: Minimize negative sum of radii
        def objective(vars):
            r = vars[2*n:]
            return -np.sum(r)
        
        # 3. Constraints: No overlap and inside boundary
        cons = []
        
        # Boundary constraints (for each circle)
        for i in range(n):
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*n + i]})
            # x + r <= 1 => 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i] - v[2*n + i]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i + 1] - v[2*n + i]})
            # y + r <= 1 => 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i + 1] - v[2*n + i]})

        # Overlap constraints (for each pair)
        for i in range(n):
            for j in range(i + 1, n):
                # dist >= r_i + r_j
                # dist^2 - (r_i + r_j)^2 >= 0
                def constraint_dist(v, i=i, j=j):
                    x_i, y_i = v[2*i], v[2*i+1]
                    x_j, y_j = v[2*j], v[2*j+1]
                    r_i, r_j = v[2*n+i], v[2*n+j]
                    dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                    sum_r = r_i + r_j
                    return dist_sq - (sum_r**2)
                
                cons.append({'type': 'ineq', 'fun': constraint_dist})

        # 4. Run Optimization
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=cons, 
                           options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success:
                curr_centers = res.x[:2*n].reshape(n, 2)
                curr_radii = res.x[2*n:]
                curr_sum = np.sum(curr_radii)
                
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = curr_centers
                    best_radii = curr_radii
        except Exception:
            continue

    if best_centers is None:
        # Fallback to a valid random configuration if optimization fails
        best_centers = np.random.rand(n, 2)
        best_radii = np.full(n, 0.01)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, best_sum
