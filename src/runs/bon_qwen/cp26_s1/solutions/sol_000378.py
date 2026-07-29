# sol_000378 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7a0a6c4a) state=67c69110 sum of radii=2.609953 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj(vars):
    """Objective function: minimize negative sum of radii (equivalent to maximizing sum of radii)"""
    # Radii are located at indices 2, 5, 8, ... (every 3rd element starting from 2)
    return -np.sum(vars[2::3])

def constr_boundary(vars):
    """Constraint function: ensures circles are inside the unit square"""
    n = len(vars) // 3
    res = np.zeros(4 * n)
    for i in range(n):
        idx = 3 * i
        x, y, r = vars[idx], vars[idx + 1], vars[idx + 2]
        # Constraints: x >= r, x + r <= 1, y >= r, y + r <= 1
        # Reformulated as g(vars) >= 0
        res[4 * i] = x - r
        res[4 * i + 1] = 1 - (x + r)
        res[4 * i + 2] = y - r
        res[4 * i + 3] = 1 - (y + r)
    return res

def constr_overlap(vars):
    """Constraint function: ensures no overlap between circles"""
    n = len(vars) // 3
    m = n * (n - 1) // 2
    res = np.zeros(m)
    k = 0
    for i in range(n):
        xi, yi, ri = vars[3 * i], vars[3 * i + 1], vars[3 * i + 2]
        for j in range(i + 1, n):
            xj, yj, rj = vars[3 * j], vars[3 * j + 1], vars[3 * j + 2]
            dx = xi - xj
            dy = yi - yj
            # Constraint: dist^2 >= (ri + rj)^2 => dist^2 - (ri + rj)^2 >= 0
            res[k] = dx * dx + dy * dy - (ri + rj) ** 2
            k += 1
    return res

def run_packing():
    n = 26
    
    # 1. Hexagonal-like initialization
    # This pattern is dense and serves as an excellent starting point for the optimizer
    centers = []
    y_offset = 0.05
    y_step = 0.18
    for i in range(7):
        y = y_offset + i * y_step
        if y > 0.95:
            break
        if i % 2 == 0:
            xs = np.linspace(0.1, 0.9, 5)
        else:
            xs = np.linspace(0.2, 0.8, 4)
        for x in xs:
            if len(centers) >= n:
                break
            centers.append([x, y])
        if len(centers) >= n:
            break
            
    centers = np.array(centers[:n])
    
    # Add small perturbation to break symmetry and avoid potential gradient issues
    np.random.seed(42)
    centers += np.random.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    radii_init = np.ones(n) * 0.06
    
    # 2. Prepare initial variables vector [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3 * i] = centers[i, 0]
        x0[3 * i + 1] = centers[i, 1]
        x0[3 * i + 2] = radii_init[i]
        
    # 3. Define bounds and constraints
    bounds = [(0, 1)] * (3 * n)
    
    constraints = [
        {'type': 'ineq', 'fun': constr_boundary},
        {'type': 'ineq', 'fun': constr_overlap}
    ]
    
    # 4. Run optimization
    res = minimize(obj, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
    
    if res.success:
        best_vars = res.x
    else:
        # Fallback to initial guess if optimization fails
        best_vars = x0 
        
    # 5. Extract results
    centers_opt = best_vars.reshape(n, 3)[:, :2]
    radii_opt = best_vars[2::3]
    
    return centers_opt, radii_opt, np.sum(radii_opt)
