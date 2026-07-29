# sol_000014 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 04e92922) state=1e2c066b sum of radii=1.820000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(vars, n):
    """Objective function to minimize negative sum of radii"""
    radii = vars[2::3]
    return -np.sum(radii)

def boundary_constraints(vars, n):
    """Constraints for circles staying inside the unit square"""
    cs = []
    for i in range(n):
        x = vars[3*i]
        y = vars[3*i+1]
        r = vars[3*i+2]
        # x >= r  => x - r >= 0
        cs.append(x - r)
        # x <= 1 - r => 1 - r - x >= 0
        cs.append(1 - r - x)
        # y >= r => y - r >= 0
        cs.append(y - r)
        # y <= 1 - r => 1 - r - y >= 0
        cs.append(1 - r - y)
    return np.array(cs)

def overlap_constraints(vars, n):
    """Constraints for non-overlapping circles"""
    cs = []
    for i in range(n):
        xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
        for j in range(i+1, n):
            xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
            # dist^2 >= (ri + rj)^2 => dist^2 - (ri + rj)^2 >= 0
            dist_sq = (xi - xj)**2 + (yi - yj)**2
            sum_r = ri + rj
            cs.append(dist_sq - sum_r**2)
    return np.array(cs)

def obj_func_vars(vars, n_val):
    return objective_func(vars, n_val)

def bound_func_vars(vars, n_val):
    return boundary_constraints(vars, n_val)

def overlap_func_vars(vars, n_val):
    return overlap_constraints(vars, n_val)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    # Initialization: 6x5 grid points
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.08)
    
    count = 0
    for i in range(6):
        for j in range(5):
            x = (i + 0.5) / 6.0
            y = (j + 0.5) / 5.0
            centers[count] = [x, y]
            count += 1
            if count == n:
                break
        if count == n:
            break
            
    # Add noise to break symmetry
    centers += rng.uniform(-0.01, 0.01, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    
    x0 = np.concatenate([centers.flatten(), radii])
    
    bounds = []
    for _ in range(n):
        bounds.extend([(0, 1), (0, 1), (1e-5, 0.5)])
        
    cons = []
    cons.append({'type': 'ineq', 'fun': bound_func_vars, 'args': (n,)})
    cons.append({'type': 'ineq', 'fun': overlap_func_vars, 'args': (n,)})
    
    try:
        # Run optimization
        res = minimize(obj_func_vars, x0, args=(n,), method='SLSQP', bounds=bounds, constraints=cons, 
                       options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False})
        final_vars = res.x
    except Exception:
        # Fallback to initial configuration
        final_vars = x0
        
    centers_out = np.array([[final_vars[3*i], final_vars[3*i+1]] for i in range(n)])
    radii_out = np.array([final_vars[3*i+2] for i in range(n)])
    radii_out = np.maximum(radii_out, 1e-6)
    
    # Validity check
    valid = True
    # Boundary check
    for i in range(n):
        x, y = centers_out[i]
        r = radii_out[i]
        if x < r - 1e-12 or x > 1 - r + 1e-12 or y < r - 1e-12 or y > 1 - r + 1e-12:
            valid = False
            break
    # Overlap check
    if valid:
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers_out[i] - centers_out[j])**2))
                if dist < radii_out[i] + radii_out[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        # Fallback to a safe grid packing
        r_safe = 0.07
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([(i+0.5)/6.0, (j+0.5)/5.0])
                if len(pts) == 26: break
            if len(pts) == 26: break
        centers_out = np.array(pts)
        radii_out = np.full(n, r_safe)
        
    return centers_out, radii_out, np.sum(radii_out)
