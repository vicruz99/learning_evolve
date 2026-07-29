# sol_000055 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 26e3ad40) state=ded93309 sum of radii=2.620919 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < -1e-12:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def run_packing():
    n = 26
    
    # Function to convert params to centers and radii
    def unpack(params):
        centers = params[:2*n].reshape(n, 2)
        radii = params[2*n:]
        return centers, radii

    # Objective: Maximize sum of radii (Minimize negative sum)
    def objective(params):
        centers, radii = unpack(params)
        return -np.sum(radii)

    # Constraints
    def make_constraints(n):
        constraints = []
        for i in range(n):
            # Wall constraints
            # x - r >= 0  => params[2*i] - params[2*n + i] >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i: p[2*i] - p[2*n + i]
            })
            # x + r <= 1  => 1 - (p[2*i] + p[2*n + i]) >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i: 1 - (p[2*i] + p[2*n + i])
            })
            # y - r >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i: p[2*i + 1] - p[2*n + i]
            })
            # y + r <= 1
            constraints.append({
                'type': 'ineq',
                'fun': lambda p, i=i: 1 - (p[2*i + 1] + p[2*n + i])
            })
        
        # Non-overlap constraints
        # dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda p, i=i, j=j: \
                        (p[2*i] - p[2*j])**2 + (p[2*i + 1] - p[2*j + 1])**2 - \
                        (p[2*n + i] + p[2*n + j])**2
                })
        return constraints

    constraints = make_constraints(n)
    
    # Bounds
    bounds = []
    for _ in range(2 * n): # x, y
        bounds.append((0, 1))
    for _ in range(n): # r
        bounds.append((0, 1))

    best_sum = 0
    best_params = None

    # Strategy 1: Grid initialization (5x5 + 1)
    # Start with valid packing of radius 0.09
    centers_init = np.zeros((n, 2))
    radii_init = np.full(n, 0.09)
    idx = 0
    for r in range(5):
        for c in range(5):
            centers_init[idx, 0] = 0.1 + c * 0.2
            centers_init[idx, 1] = 0.1 + r * 0.2
            idx += 1
    # 26th circle in a gap
    centers_init[25, 0] = 0.2
    centers_init[25, 1] = 0.2
    radii_init[25] = 0.01 # Small radius to start
    
    params_grid = np.concatenate([centers_init.flatten(), radii_init])
    
    # Strategy 2: Hexagonal initialization
    centers_hex = np.zeros((n, 2))
    radii_hex = np.full(n, 0.095) # Try slightly larger
    # 5 rows: 5, 6, 5, 6, 4 ? Total 26.
    # Or just place them in a staggered grid
    idx = 0
    row_y = 0.1
    for r_idx in range(5):
        is_shifted = (r_idx % 2 == 1)
        # Number of circles in row
        if is_shifted:
            n_circles = 6
            start_x = 0.09 # Shifted to fit 6
        else:
            n_circles = 5
            start_x = 0.1
            
        # Adjust spacing to fit
        if n_circles == 6:
            spacing = 0.16 # 6 circles width approx 1.0
        else:
            spacing = 0.2
            
        for c in range(n_circles):
            if idx < n:
                x = start_x + c * spacing
                y = row_y
                centers_hex[idx] = [x, y]
                idx += 1
        row_y += 0.16 # Vertical spacing
        
    # Fill remaining if any
    while idx < n:
        centers_hex[idx] = [0.5, 0.5]
        radii_hex[idx] = 0.01
        idx += 1
        
    params_hex = np.concatenate([centers_hex.flatten(), radii_hex])

    # Strategy 3: Random initialization
    rng = np.random.default_rng(42)
    centers_rand = rng.uniform(0.1, 0.9, size=(n, 2))
    radii_rand = np.full(n, 0.08)
    params_rand = np.concatenate([centers_rand.flatten(), radii_rand])

    initializations = [params_grid, params_hex, params_rand]

    for params in initializations:
        try:
            res = minimize(objective, params, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-8})
            if res.success or res.fun < -best_sum: # res.fun is negative sum
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_params = res.x
        except Exception as e:
            pass

    # If no good solution found, try a second pass with best_params as start
    if best_params is not None:
        try:
            res2 = minimize(objective, best_params, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 2000})
            if -res2.fun > best_sum:
                best_sum = -res2.fun
                best_params = res2.x
        except:
            pass

    centers = unpack(best_params)[0]
    radii = unpack(best_params)[1]
    
    # Ensure radii are non-negative and clipped
    radii = np.maximum(radii, 0)
    
    # Final validation check
    if validate_packing(centers, radii):
        return centers, radii, np.sum(radii)
    else:
        # Fallback to a simple valid grid
        centers = np.zeros((n, 2))
        radii = np.full(n, 0.1)
        idx = 0
        for r in range(5):
            for c in range(5):
                centers[idx] = [0.1 + c * 0.2, 0.1 + r * 0.2]
                idx += 1
        # 26th circle small
        centers[25] = [0.5, 0.5]
        radii[25] = 0.001
        return centers, radii, np.sum(radii)
