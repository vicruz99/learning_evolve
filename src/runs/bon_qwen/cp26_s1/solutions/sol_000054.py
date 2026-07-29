# sol_000054 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 50e7db78) state=096920f1 sum of radii=2.575799 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    # Radii are at indices 2, 5, 8, ...
    return -np.sum(vars[2::3])

def pair_constraint(vars, i, j):
    """Constraint: distance between centers >= sum of radii."""
    xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
    xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
    dist_sq = (xi - xj)**2 + (yi - yj)**2
    return dist_sq - (ri + rj)**2

def boundary_constraint(vars, i, t):
    """Constraint: circle i stays inside [0,1]x[0,1]."""
    x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
    if t == 0: return x - r
    if t == 1: return y - r
    if t == 2: return 1.0 - x - r
    return 1.0 - y - r

def run_packing():
    n = 26
    
    # 1. Initialize with a hexagonal grid pattern
    centers = []
    s = 0.22  # Initial spacing
    y = 0.15
    while len(centers) < n:
        x = 0.15 + (len(centers) % 2) * (s / 2)
        while x < 0.95 and len(centers) < n:
            centers.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
    
    centers = np.array(centers[:n])
    radii = np.full(n, 0.05)
    
    # Flatten to vector [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # 2. Define bounds
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    # 3. Define constraints
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': pair_constraint, 'args': (i, j)})
        for t in range(4):
            cons.append({'type': 'ineq', 'fun': boundary_constraint, 'args': (i, t)})

    # 4. Run optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 5000, 'ftol': 1e-12})
                   
    # 5. Extract and return results
    centers_out = res.x.reshape((n, 3))[:, :2]
    radii_out = res.x.reshape((n, 3))[:, 2]
    return centers_out, radii_out, np.sum(radii_out)
