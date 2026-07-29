# sol_000030 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 55285a70) state=fbc1f7e6 sum of radii=2.498640 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(x, n):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(x[2::3])

def constraint_func(x, n):
    """Constraints: boundary and non-overlap. Returns array of values >= 0."""
    n_constraints = 4 * n + n * (n - 1) // 2
    res = np.zeros(n_constraints)
    idx = 0
    
    # Boundary constraints
    for i in range(n):
        b = 3 * i
        xi, yi, ri = x[b], x[b+1], x[b+2]
        res[idx]   = xi - ri          # x >= r
        res[idx+1] = 1 - xi - ri      # x <= 1-r
        res[idx+2] = yi - ri          # y >= r
        res[idx+3] = 1 - yi - ri      # y <= 1-r
        idx += 4
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = 3 * i, 3 * j
            dx = x[bi] - x[bj]
            dy = x[bi+1] - x[bj+1]
            r_sum = x[bi+2] + x[bj+2]
            res[idx] = dx**2 + dy**2 - r_sum**2
            idx += 1
    return res

def run_packing():
    n = 26
    
    # 1. Initial configuration: Hexagonal-like grid
    centers_init = []
    row_counts = [6, 5, 6, 5, 4]  # Sums to 26
    y = 0.15
    for i, cnt in enumerate(row_counts):
        x = 0.15 + (0.05 if i % 2 == 1 else 0.0)
        for _ in range(cnt):
            centers_init.append([x, y])
            x += 0.14
        y += 0.18
        
    # Flatten to [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.array([val for c in centers_init for val in [*c, 0.04]])
    
    # Variable bounds
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # Constraint definition
    cons = {'type': 'ineq', 'fun': constraint_func, 'args': (n,)}
    
    # 2. Optimize
    res = minimize(
        objective_func, 
        x0, 
        args=(n,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False}
    )
    
    # 3. Extract and format results
    centers = np.column_stack((res.x[::3], res.x[1::3]))
    radii = np.maximum(res.x[2::3], 0.0)
    
    # Ensure strict boundary compliance after optimization
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
