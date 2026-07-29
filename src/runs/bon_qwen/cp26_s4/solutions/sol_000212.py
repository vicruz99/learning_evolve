# sol_000212 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 722eaafb) state=df52446e sum of radii=1.040000 correctness=1.0
# stdout(first 200): Optimization failed for config 0: arrays used as indices must be of integer (or boolean) type Optimization failed for config 1: arrays used as indices must be of integer (or boolean) type Optimization
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import itertools

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # Helper to define constraints for scipy
    def define_constraints(params, n):
        constraints = []
        
        # Boundary constraints:
        # r <= x <= 1-r  =>  x - r >= 0, x + r <= 1
        # r <= y <= 1-r  =>  y - r >= 0, y + r <= 1
        
        for i in range(n):
            # x - r >= 0
            def bound_x_min(i=i, params=params):
                # params layout: x1, y1, r1, x2, y2, r2, ...
                # index for x_i is 3*i, y_i is 3*i+1, r_i is 3*i+2
                x = params[3*i]
                r = params[3*i+2]
                return x - r
            constraints.append({'type': 'ineq', 'fun': bound_x_min})
            
            # 1 - x - r >= 0
            def bound_x_max(i=i, params=params):
                x = params[3*i]
                r = params[3*i+2]
                return 1 - x - r
            constraints.append({'type': 'ineq', 'fun': bound_x_max})
            
            # y - r >= 0
            def bound_y_min(i=i, params=params):
                y = params[3*i+1]
                r = params[3*i+2]
                return y - r
            constraints.append({'type': 'ineq', 'fun': bound_y_min})
            
            # 1 - y - r >= 0
            def bound_y_max(i=i, params=params):
                y = params[3*i+1]
                r = params[3*i+2]
                return 1 - y - r
            constraints.append({'type': 'ineq', 'fun': bound_y_max})
            
            # r >= 0
            def bound_r_pos(i=i, params=params):
                return params[3*i+2]
            constraints.append({'type': 'ineq', 'fun': bound_r_pos})

        # Non-overlap constraints:
        # dist(c_i, c_j) >= r_i + r_j
        # sqrt((xi-xj)^2 + (yi-yj)^2) - ri - rj >= 0
        for i in range(n):
            for j in range(i + 1, n):
                def dist_constraint(i=i, j=j, params=params):
                    xi, yi, ri = params[3*i], params[3*i+1], params[3*i+2]
                    xj, yj, rj = params[3*j], params[3*j+1], params[3*j+2]
                    
                    dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                    return dist - (ri + rj)
                constraints.append({'type': 'ineq', 'fun': dist_constraint})
        
        return constraints

    # Objective function: minimize -sum(radii)
    def objective(params, n):
        radii = params[2::3] # every 3rd element starting from index 2
        return -np.sum(radii)

    # Generate initial configurations
    initial_configs = []

    # 1. Grid initialization (5x5 grid is 25, so 5x6 or distorted)
    # Let's try a 5x6 grid spaced out, but we only need 26.
    # Actually, a dense grid might be a good start.
    # Let's create a 5x6 grid and take first 26 points?
    # Or a square-ish distribution.
    
    # Config 1: Perturbed Grid
    # 5 rows, roughly 5-6 cols.
    # Let's place them in a grid 0..1
    rows = 5
    cols = 6 # 5*6 = 30 > 26. We can pick 26 best spots or just first 26.
    # Better: 5 rows of 5 and 1 extra?
    # Let's just do a regular grid of size sqrt(26) approx 5.1
    # Let's try 5 rows, 6 columns, take first 26.
    grid_x = np.linspace(0.1, 0.9, 6) # 6 points
    grid_y = np.linspace(0.1, 0.9, 5) # 5 points
    
    # Generate all grid points
    points = []
    for y in grid_y:
        for x in grid_x:
            points.append((x, y))
    
    # We need 26 points. We have 30.
    # Let's select a subset that looks balanced.
    # Just take first 26.
    init_points = points[:26]
    
    params = []
    for x, y in init_points:
        # Initial radius: estimate based on neighbors? 
        # Just start with 0.05
        params.extend([x, y, 0.05])
    initial_configs.append(np.array(params))

    # Config 2: Random initialization
    np.random.seed(42)
    params_rand = []
    for _ in range(26):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = np.random.uniform(0.02, 0.08)
        params_rand.extend([x, y, r])
    initial_configs.append(np.array(params_rand))

    # Config 3: Hexagonal-like packing initialization
    # Try to pack in rows
    params_hex = []
    count = 0
    y_pos = 0.15
    while count < 26:
        x_pos = 0.15
        while count < 26 and x_pos <= 0.9:
            # Add circle
            r = 0.08 # initial guess
            params_hex.extend([x_pos, y_pos, r])
            count += 1
            x_pos += 0.2 # spacing
        y_pos += 0.2 # row spacing
        if y_pos > 0.85: break # avoid too many rows
    # If we didn't reach 26, this loop logic is simple but might fail if bounds tight
    # Let's rely on the optimizer to fix it, but ensure we have 26
    # Actually the loop above might produce fewer if width is small.
    # Let's force 26 points.
    if len(params_hex) < 26 * 3:
        # Fallback to random
        initial_configs.append(np.array(params_rand))
    else:
        initial_configs.append(np.array(params_hex[:26*3]))

    # Config 4: Another random seed
    np.random.seed(123)
    params_rand2 = []
    for _ in range(26):
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        r = np.random.uniform(0.02, 0.08)
        params_rand2.extend([x, y, r])
    initial_configs.append(np.array(params_rand2))

    # Optimization loop
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * 26
    constraints = define_constraints(np.zeros(26*3), 26)

    for i, x0 in enumerate(initial_configs):
        if len(x0) != 26 * 3:
            continue
            
        # Scale x0 to fit bounds if necessary (already done in generation mostly)
        # Ensure r is not too large initially to avoid infeasibility
        for k in range(0, len(x0), 3):
            if x0[k+2] > 0.5: x0[k+2] = 0.5
            if x0[k+2] < 0: x0[k+2] = 0.0
            
        try:
            # Use SLSQP
            res = minimize(
                objective, 
                x0, 
                args=(26,), 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-9, 'disp': False}
            )
            
            if res.success or (not res.success and res.fun < -2.0): # Check if we got a decent result
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    # Extract solution
                    best_params = res.x
                    centers = np.array([[best_params[3*j], best_params[3*j+1]] for j in range(26)])
                    radii = np.array([best_params[3*j+2] for j in range(26)])
                    
                    # Validate manually before storing as best
                    # Quick check
                    valid = True
                    for j in range(26):
                        x, y = centers[j]
                        r = radii[j]
                        if x < -1e-9 or x > 1+1e-9 or y < -1e-9 or y > 1+1e-9:
                            valid = False; break
                        if r < -1e-9:
                            valid = False; break
                        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                            valid = False; break
                        for k in range(j+1, 26):
                            dist = np.sqrt((centers[j][0]-centers[k][0])**2 + (centers[j][1]-centers[k][1])**2)
                            if dist < radii[j] + radii[k] - 1e-9:
                                valid = False; break
                        if not valid: break
                    
                    if valid:
                        best_centers = centers
                        best_radii = radii
        except Exception as e:
            print(f"Optimization failed for config {i}: {e}")
            continue

    # If no valid solution found via optimization (unlikely), return a simple valid one
    if best_centers is None:
        # Fallback: 26 small circles in a grid
        centers = np.zeros((26, 2))
        radii = np.full(26, 0.05)
        idx = 0
        for i in range(5):
            for j in range(6):
                if idx < 26:
                    centers[idx] = [0.15 + j*0.15, 0.15 + i*0.15]
                    radii[idx] = 0.04 # small enough to not overlap
                    idx += 1
        best_centers = centers
        best_radii = radii
        best_sum_radii = np.sum(radii)

    return best_centers, best_radii, best_sum_radii
