# sol_000067 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cae61cda) state=a28eaee3 sum of radii=2.613222 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

def objective(v, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def make_bound_constraint(i):
    """Factory for boundary constraint functions for circle i."""
    def func(v):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        # Returns array of 4 values, all must be >= 0
        return np.array([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    return func

def make_dist_constraint(i, j):
    """Factory for distance constraint functions for pair (i, j)."""
    def func(v):
        xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
        xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
        # dist^2 >= (r_i + r_j)^2  =>  dist^2 - (r_i + r_j)^2 >= 0
        return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
    return func

def run_packing():
    n = 26
    
    # 1. Initialization on a hexagonal grid
    centers = np.zeros((n, 2))
    idx = 0
    y = 0.12
    row = 0
    while idx < n:
        x = 0.12 + (row % 2) * 0.09
        while x <= 0.88 and idx < n:
            centers[idx] = [x, y]
            idx += 1
            x += 0.17
        y += 0.147
        row += 1
        
    # Fallback fill if loop terminates early (should not happen with params)
    while idx < n:
        centers[idx] = centers[idx-1] + [0.02, 0.02]
        idx += 1
        
    radii0 = np.full(n, 0.04)
    
    # Flatten to optimization vector: [x1, y1, r1, x2, y2, r2, ...]
    vars0 = np.zeros(3*n)
    for i in range(n):
        vars0[3*i] = centers[i, 0]
        vars0[3*i+1] = centers[i, 1]
        vars0[3*i+2] = radii0[i]
        
    # Bounds: coordinates in [0, 1], radii in [epsilon, 0.5]
    bounds = [(0, 1)] * (3*n)
    for i in range(n):
        bounds[3*i+2] = (1e-6, 0.5)
        
    # 2. Build constraints
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': make_bound_constraint(i)})
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': make_dist_constraint(i, j)})
            
    # 3. Optimize
    res = minimize(
        objective, 
        vars0, 
        args=(n,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        options={'maxiter': 8000, 'ftol': 1e-10, 'disp': False}
    )
    
    # 4. Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        final_centers[i] = [res.x[3*i], res.x[3*i+1]]
        final_radii[i] = max(0.0, res.x[3*i+2])
        
    return final_centers, final_radii, np.sum(final_radii)
