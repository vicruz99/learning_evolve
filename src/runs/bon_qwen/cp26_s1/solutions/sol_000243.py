# sol_000243 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a213d118) state=4ff027fb sum of radii=2.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_overlap_constraints(centers, radii):
    """
    Calculate the distance margin for all pairs.
    Returns an array of size N*(N-1)/2 where each value is dist - (r_i + r_j).
    We want these to be >= 0.
    """
    n = len(centers)
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = radii[i] + radii[j]
            constraints.append(dist - min_dist)
    return np.array(constraints)

def obj_func(params, n):
    """Objective function: minimize negative sum of radii."""
    radii = params[2::3] # r1, r2, ...
    return -np.sum(radii)

def constraint_func(params, n):
    """Returns the values of the non-overlap constraints."""
    centers = np.column_stack((params[0::3], params[1::3]))
    radii = params[2::3]
    return get_overlap_constraints(centers, radii)

def run_packing():
    n = 26
    best_sum_radii = -np.inf
    best_params = None
    
    # Bounds: [0, 1] for x, y. [0, 0.5] for r (cannot exceed 0.5 in unit square)
    bounds = [(0, 1) for _ in range(n)] + [(0, 1) for _ in range(n)] + [(0, 0.5) for _ in range(n)]
    
    # Define constraints for the solver
    # Inequality constraints: g(x) >= 0
    cons = ({'type': 'ineq', 'fun': lambda p: constraint_func(p, n)})

    # Initial guesses
    initial_guesses = []
    
    # 1. Hexagonal lattice patterns with different shifts/scaling
    for _ in range(5):
        params = np.zeros(3 * n)
        r_est = 0.08 # Start small to ensure validity
        idx = 0
        row = 0
        col = 0
        while idx < n:
            # Hexagonal coordinates
            x = r_est + col * (2 * r_est)
            if row % 2 == 1:
                x += r_est
            y = r_est + row * (r_est * np.sqrt(3))
            
            if x <= 1 - r_est and y <= 1 - r_est:
                params[3*idx] = x + np.random.uniform(-0.01, 0.01)
                params[3*idx + 1] = y + np.random.uniform(-0.01, 0.01)
                params[3*idx + 2] = r_est
                idx += 1
                col += 1
            else:
                row += 1
                col = 0
        
        # Pad if fewer than n circles generated (should not happen with small r)
        while idx < n:
            params[3*idx] = 0.5 + np.random.uniform(-0.4, 0.4)
            params[3*idx + 1] = 0.5 + np.random.uniform(-0.4, 0.4)
            params[3*idx + 2] = 0.01
            idx += 1
        initial_guesses.append(params)

    # 2. Random placements
    for _ in range(3):
        params = np.zeros(3 * n)
        for i in range(n):
            r = 0.05 + np.random.uniform(0, 0.02)
            x = np.random.uniform(r, 1-r)
            y = np.random.uniform(r, 1-r)
            params[3*i] = x
            params[3*i + 1] = y
            params[3*i + 2] = r
        initial_guesses.append(params)

    # 3. Grid patterns
    for _ in range(2):
        params = np.zeros(3 * n)
        r_est = 0.06
        idx = 0
        step = 1.0 / 5.0
        for i in range(6):
            for j in range(6):
                if idx < n:
                    x = step/2 + i * step + np.random.uniform(-0.02, 0.02)
                    y = step/2 + j * step + np.random.uniform(-0.02, 0.02)
                    # Ensure within bounds
                    x = np.clip(x, r_est, 1-r_est)
                    y = np.clip(y, r_est, 1-r_est)
                    params[3*idx] = x
                    params[3*idx + 1] = y
                    params[3*idx + 2] = r_est
                    idx += 1
        initial_guesses.append(params)

    # Run optimization
    options = {'maxiter': 500, 'ftol': 1e-9}

    for i, x0 in enumerate(initial_guesses):
        try:
            # Try to optimize
            res = minimize(obj_func, x0, args=(n,), method='SLSQP', 
                           bounds=bounds, constraints=cons, options=options)
            
            if res.success or res.fun < best_sum_radii * -1: # Minimize -sum -> max sum
                current_sum = -res.fun
                # Validate manually to be safe against solver quirks
                centers = np.column_stack((res.x[0::3], res.x[1::3]))
                radii = res.x[2::3]
                
                # Basic validation
                valid = True
                for k in range(n):
                    if radii[k] < 0: valid = False
                    if centers[k][0] - radii[k] < -1e-6 or centers[k][0] + radii[k] > 1 + 1e-6: valid = False
                    if centers[k][1] - radii[k] < -1e-6 or centers[k][1] + radii[k] > 1 + 1e-6: valid = False
                
                if valid:
                     for k in range(n):
                        for l in range(k+1, n):
                            d = np.linalg.norm(centers[k] - centers[l])
                            if d < radii[k] + radii[l] - 1e-6:
                                valid = False
                                break
                        if not valid: break
                
                if valid and current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_params = res.x.copy()
                    # print(f"Run {i}: Sum = {current_sum:.5f}")
        except Exception as e:
            pass

    if best_params is None:
        # Fallback to a simple grid if optimization failed
        params = np.zeros(3 * n)
        r = 0.08
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < n:
                    params[3*idx] = 0.1 + i * 0.2
                    params[3*idx + 1] = 0.1 + j * 0.2
                    params[3*idx + 2] = r
                    idx += 1
        best_params = params
        best_sum_radii = 26 * r

    centers = np.column_stack((best_params[0::3], best_params[1::3]))
    radii = best_params[2::3]
    
    return centers, radii, np.sum(radii)
