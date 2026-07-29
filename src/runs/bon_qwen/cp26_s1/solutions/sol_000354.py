# sol_000354 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b8d6b6a1) state=fa91ca00 sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_initial_hexagonal_centers(n_circles):
    """
    Generates a hexagonal grid arrangement of points for initial packing.
    """
    centers_list = []
    r_est = 0.1  # Initial radius estimate
    h = r_est * np.sqrt(3) # Vertical spacing in hex grid
    y = r_est
    row_type = 0  # 0 for 5 circles, 1 for 4 circles
    
    # Generate rows until we have enough points or exceed height
    while y <= 1.0 + 1e-5 and len(centers_list) < n_circles:
        if row_type == 0:
            # 5 circles row: x at 0.1, 0.3, 0.5, 0.7, 0.9
            x_start = r_est
            for k in range(5):
                if len(centers_list) >= n_circles:
                    break
                x = x_start + k * (2 * r_est)
                centers_list.append([x, y])
        else:
            # 4 circles row (offset): x at 0.2, 0.4, 0.6, 0.8
            x_start = r_est + r_est 
            for k in range(4):
                if len(centers_list) >= n_circles:
                    break
                x = x_start + k * (2 * r_est)
                centers_list.append([x, y])
        
        y += h
        row_type = 1 - row_type
        
    return np.array(centers_list[:n_circles])

def calculate_dist_constraint(x_vars, n, i, j):
    """
    Calculates the non-overlap constraint between circle i and circle j.
    Constraint: dist >= r_i + r_j  =>  dist - (r_i + r_j) >= 0
    """
    xi, yi, ri = x_vars[3*i], x_vars[3*i+1], x_vars[3*i+2]
    xj, yj, rj = x_vars[3*j], x_vars[3*j+1], x_vars[3*j+2]
    
    # Use a small epsilon to avoid sqrt(0) issues or division by zero if needed, 
    # though here simple sqrt is fine.
    dist_sq = (xi - xj)**2 + (yi - yj)**2
    dist = np.sqrt(dist_sq)
    
    return dist - (ri + rj)

def calculate_boundary_constraint_x_min(x_vars, n, i):
    """Constraint: x_i >= r_i => x_i - r_i >= 0"""
    return x_vars[3*i] - x_vars[3*i+2]

def calculate_boundary_constraint_x_max(x_vars, n, i):
    """Constraint: 1 - x_i >= r_i => 1 - x_i - r_i >= 0"""
    return 1 - x_vars[3*i] - x_vars[3*i+2]

def calculate_boundary_constraint_y_min(x_vars, n, i):
    """Constraint: y_i >= r_i => y_i - r_i >= 0"""
    return x_vars[3*i+1] - x_vars[3*i+2]

def calculate_boundary_constraint_y_max(x_vars, n, i):
    """Constraint: 1 - y_i >= r_i => 1 - y_i - r_i >= 0"""
    return 1 - x_vars[3*i+1] - x_vars[3*i+2]

def objective_function(x_vars, n):
    """
    Objective: Maximize sum of radii.
    Optimization minimizes, so we return negative sum.
    """
    total_radius = sum(x_vars[3*i+2] for i in range(n))
    return -total_radius

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize centers with hexagonal packing
    centers_init = get_initial_hexagonal_centers(n)
    radii_init = np.full(n, 0.1)
    
    # 2. Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    # 3. Define Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # 4. Define Constraints
    cons = []
    
    # Boundary constraints (4 per circle)
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: calculate_boundary_constraint_x_min(x, n, idx)})
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: calculate_boundary_constraint_x_max(x, n, idx)})
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: calculate_boundary_constraint_y_min(x, n, idx)})
        cons.append({'type': 'ineq', 'fun': lambda x, idx=i: calculate_boundary_constraint_y_max(x, n, idx)})
        
    # Non-overlap constraints (between all pairs)
    # We add constraints for all pairs to ensure validity.
    # In a dense packing, many of these will be active (tight).
    for i in range(n):
        for j in range(i + 1, n):
            # To avoid closure variable capture issues, we pass i, j as default args
            # However, the prompt forbids lambdas. So we use a helper or pass indices.
            # Since we cannot use lambdas, we define the function logic inside the loop 
            # but we need to pass it. We can use a factory function or just rely on the 
            # fact that the optimizer calls the function. 
            # Actually, standard practice without lambda is to create a distinct function 
            # or use a closure properly. But prompt says "no lambdas".
            # I will use the top-level helper and pass indices. 
            # But 'fun' requires a callable. 
            # I'll create a list of constraints using a helper that captures i, j.
            # Wait, I can't use lambda. I can define a class or use a function that 
            # takes the closure variables. But 'fun' takes (x).
            # So I need a function f(x) that knows i and j.
            # I can use a partial function from functools, or just a local function?
            # "Make all helper functions top level".
            # So I cannot define functions inside run_packing.
            # I must use top-level functions.
            # I can pass (i, j) via a wrapper? No lambdas.
            # I can use a tuple constraint? No.
            # I can use `scipy.optimize` constraint with args? 
            # `fun` can take (x, *args). 
            # So: {'type': 'ineq', 'fun': calculate_dist_constraint, 'args': (n, i, j)}
            pass 

    # Re-building constraints list properly without lambdas
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': calculate_boundary_constraint_x_min, 'args': (n, i)})
        cons.append({'type': 'ineq', 'fun': calculate_boundary_constraint_x_max, 'args': (n, i)})
        cons.append({'type': 'ineq', 'fun': calculate_boundary_constraint_y_min, 'args': (n, i)})
        cons.append({'type': 'ineq', 'fun': calculate_boundary_constraint_y_max, 'args': (n, i)})
        
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': calculate_dist_constraint, 'args': (n, i, j)})

    # 5. Optimize
    result = minimize(
        objective_function, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons,
        args=(n,),
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    # 6. Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = result.x[3*i]
        final_centers[i, 1] = result.x[3*i+1]
        final_radii[i] = result.x[3*i+2]
        
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
