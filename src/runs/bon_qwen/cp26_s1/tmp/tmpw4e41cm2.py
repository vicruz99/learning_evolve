import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization
    # Start with a 5x5 grid pattern to ensure a valid initial state, 
    # but with slightly smaller radii to allow room for movement and growth.
    # 5x5 grid spacing is 0.2, so radius 0.1 fits. We start with 0.05 to be safe and loose.
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    idx = 0
    # Create a 5x5 grid
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + i * 0.2, 0.1 + j * 0.2]
            radii[idx] = 0.05
            idx += 1
    # Add the 26th circle in a gap (e.g., center of a hole)
    # Hole at (0.2, 0.2) is surrounded by (0.1,0.1), (0.3,0.1), (0.1,0.3), (0.3,0.3)
    # Distance to centers is sqrt(0.1^2 + 0.1^2) ~ 0.1414. 
    # Available radius ~ 0.1414 - 0.05 = 0.0914. Let's place it small initially.
    centers[idx] = [0.2, 0.2]
    radii[idx] = 0.02

    # Flatten variables for optimization: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3 * i] = centers[i, 0]
        x0[3 * i + 1] = centers[i, 1]
        x0[3 * i + 2] = radii[i]

    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)])

    # Objective: Maximize sum of radii -> Minimize negative sum
    def objective(vars_flat):
        r = vars_flat[2::3]
        return -np.sum(r)

    # Constraints
    def make_constraints(vars_flat):
        constraints_list = []
        
        # Extract centers and radii
        xs = vars_flat[0::3]
        ys = vars_flat[1::3]
        rs = vars_flat[2::3]
        
        # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        # x - r >= 0  =>  x - r
        for i in range(n):
            constraints_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
            constraints_list.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
            constraints_list.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
            constraints_list.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})

        # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                def make_dist_constraint(idx_i, idx_j):
                    def dist_constraint(v):
                        x_i, y_i, r_i = v[3*idx_i], v[3*idx_i+1], v[3*idx_i+2]
                        x_j, y_j, r_j = v[3*idx_j], v[3*idx_j+1], v[3*idx_j+2]
                        dist_sq = (x_i - x_j)**2 + (y_i - y_j)**2
                        rad_sum_sq = (r_i + r_j)**2
                        return dist_sq - rad_sum_sq
                    return dist_constraint
                
                constraints_list.append({
                    'type': 'ineq', 
                    'fun': make_dist_constraint(i, j)
                })
                
        return constraints_list

    # We construct constraints once based on structure, but the functions capture indices.
    # However, SLSQP expects a list of dicts. We need to generate them dynamically or use a wrapper.
    # Generating them inside a function called by minimize is not possible directly for the list itself,
    # but we can return the list.
    
    # To avoid recreating the list every evaluation (which is expensive), 
    # we can define a class or use a closure, but SLSQP calls fun(vars).
    # We will construct the list of constraint functions once.
    
    constraints = []
    # Boundary constraints
    for i in range(n):
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i] - v[3*i+2]})
        # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1 - v[3*i+1] - v[3*i+2]})

    # Pairwise constraints
    for i in range(n):
        for j in range(i + 1, n):
            # Closure to capture i and j
            def dist_con(idx_i, idx_j):
                def func(v):
                    x_i, y_i, r_i = v[3*idx_i], v[3*idx_i+1], v[3*idx_i+2]
                    x_j, y_j, r_j = v[3*idx_j], v[3*idx_j+1], v[3*idx_j+2]
                    return (x_i - x_j)**2 + (y_i - y_j)**2 - (r_i + r_j)**2
                return func
            constraints.append({'type': 'ineq', 'fun': dist_con(i, j)})

    # Run optimization
    # We might need to run it a few times or use a good method. 
    # SLSQP is robust for this size.
    
    # To improve chances of finding global optimum, we can run multiple restarts with slightly perturbed starts
    # or just rely on the solver. Given the complexity, let's try one strong run.
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False})

    if res.success:
        best_vars = res.x
    else:
        # Fallback to initial if failed, or try to use result anyway
        best_vars = res.x

    # Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = best_vars[3*i]
        final_centers[i, 1] = best_vars[3*i+1]
        final_radii[i] = best_vars[3*i+2]

    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# Helper function to run packing if needed outside, but run_packing is the entry point