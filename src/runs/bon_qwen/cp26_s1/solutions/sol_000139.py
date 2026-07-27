# sol_000139 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4c86e033) state=f9e97e48 sum of radii=2.610742 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_):
    """
    Objective function: negative sum of radii (to be minimized).
    vars_ is a flattened array: [x1, y1, r1, x2, y2, r2, ...]
    """
    radii = vars_[2::3]
    return -np.sum(radii)

def compute_constraints(vars_):
    """
    Compute constraint violations.
    Returns an array of constraint values >= 0.
    """
    n = len(vars_) // 3
    
    centers_x = vars_[0::3]
    centers_y = vars_[1::3]
    radii = vars_[2::3]
    
    constraints = []
    
    # Boundary constraints
    # x - r >= 0
    constraints.extend(centers_x - radii)
    # 1 - x - r >= 0
    constraints.extend(1 - centers_x - radii)
    # y - r >= 0
    constraints.extend(centers_y - radii)
    # 1 - y - r >= 0
    constraints.extend(1 - centers_y - radii)
    
    # Non-overlap constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers_x[i] - centers_x[j]
            dy = centers_y[i] - centers_y[j]
            dist_sq = dx*dx + dy*dy
            sum_r = radii[i] + radii[j]
            constraints.append(dist_sq - sum_r*sum_r)
            
    return np.array(constraints)

def get_grid_init(n):
    """
    Generate a grid-like initial configuration.
    """
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.01)
    
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count < n:
                x = (c + 0.5) / cols
                y = (r + 0.5) / rows
                centers[count] = [x, y]
                count += 1
            else:
                break
    return centers, radii

def get_hex_init(n):
    """
    Generate a hexagonal-like initial configuration.
    """
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.01)
    
    count = 0
    row = 0
    y = 0.1
    
    while count < n:
        step = 0.2
        if row % 2 == 0:
            start_x = 0.1
        else:
            start_x = 0.2
            
        x = start_x
        while x <= 0.9 and count < n:
            centers[count] = [x, y]
            count += 1
            x += step
        
        y += 0.15 
        row += 1
        
    return centers, radii

def run_packing():
    n = 26
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, 0.5))
        
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_sum_r = -1.0
    best_centers = None
    best_radii = None
    
    # Strategy 1: Grid Init
    centers_init, radii_init = get_grid_init(n)
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False})
        if res.success:
            current_r = np.zeros(n)
            current_c = np.zeros((n, 2))
            for i in range(n):
                current_c[i] = res.x[3*i : 3*i+2]
                current_r[i] = res.x[3*i+2]
            s = np.sum(current_r)
            if s > best_sum_r:
                best_sum_r = s
                best_centers = current_c
                best_radii = current_r
    except Exception:
        pass

    # Strategy 2: Hex Init
    centers_init, radii_init = get_hex_init(n)
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False})
        if res.success:
            current_r = np.zeros(n)
            current_c = np.zeros((n, 2))
            for i in range(n):
                current_c[i] = res.x[3*i : 3*i+2]
                current_r[i] = res.x[3*i+2]
            s = np.sum(current_r)
            if s > best_sum_r:
                best_sum_r = s
                best_centers = current_c
                best_radii = current_r
    except Exception:
        pass

    # Strategy 3: Random Inits
    np.random.seed(123)
    for _ in range(3):
        centers_init = np.random.rand(n, 2)
        radii_init = np.full(n, 0.02)
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers_init[i, 0]
            x0[3*i+1] = centers_init[i, 1]
            x0[3*i+2] = radii_init[i]
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False})
            if res.success:
                current_r = np.zeros(n)
                current_c = np.zeros((n, 2))
                for i in range(n):
                    current_c[i] = res.x[3*i : 3*i+2]
                    current_r[i] = res.x[3*i+2]
                s = np.sum(current_r)
                if s > best_sum_r:
                    best_sum_r = s
                    best_centers = current_c
                    best_radii = current_r
        except Exception:
            pass

    if best_centers is None:
        centers = np.random.rand(n, 2)
        radii = np.full(n, 0.01)
        return centers, radii, np.sum(radii)

    return best_centers, best_radii, best_sum_r
