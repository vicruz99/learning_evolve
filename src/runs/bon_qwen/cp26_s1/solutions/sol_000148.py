# sol_000148 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e234a3e4) state=bd41047d sum of radii=2.592939 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_func(x, n):
    """
    Objective function to minimize: -sum(radii)
    x is the flattened vector of [x1, y1, r1, x2, y2, r2, ...]
    """
    # Radii are at indices 2, 5, 8, ... (every 3rd element starting at 2)
    r = x[2::3]
    return -np.sum(r)

def bound_con(x, n):
    """
    Boundary constraints:
    r <= x <= 1-r  =>  x - r >= 0  AND  1 - x - r >= 0
    r <= y <= 1-r  =>  y - r >= 0  AND  1 - y - r >= 0
    """
    con = np.zeros(n * 4)
    idx = 0
    for i in range(n):
        xi = x[3*i]
        yi = x[3*i+1]
        ri = x[3*i+2]
        con[idx] = xi - ri
        con[idx+1] = 1.0 - xi - ri
        con[idx+2] = yi - ri
        con[idx+3] = 1.0 - yi - ri
        idx += 4
    return con

def pair_con(x, n):
    """
    Pairwise non-overlap constraints:
    dist(i, j)^2 >= (ri + rj)^2  =>  dist^2 - (ri + rj)^2 >= 0
    """
    constraints = []
    for i in range(n):
        xi = x[3*i]
        yi = x[3*i+1]
        ri = x[3*i+2]
        for j in range(i + 1, n):
            xj = x[3*j]
            yj = x[3*j+1]
            rj = x[3*j+2]
            dx = xi - xj
            dy = yi - yj
            dist_sq = dx*dx + dy*dy
            r_sum = ri + rj
            constraints.append(dist_sq - r_sum*r_sum)
    return np.array(constraints)

def run_packing():
    """
    Runs the packing optimization for 26 circles.
    """
    n = 26
    np.random.seed(42)
    
    # --- Initialization ---
    # Start with a 5x5 grid for 25 circles
    x_grid = np.linspace(0.1, 0.9, 5)
    y_grid = np.linspace(0.1, 0.9, 5)
    X, Y = np.meshgrid(x_grid, y_grid)
    centers = np.column_stack((X.flatten(), Y.flatten()))
    
    # Add the 26th circle. 
    # Place it near the corner (0,0) with a safe radius to avoid initial overlap.
    # Grid points are at 0.1, 0.3... Distance from (0.04, 0.04) to (0.1, 0.1) is ~0.085.
    # With r=0.04, 2r = 0.08. 0.085 > 0.08, so it's valid.
    centers = np.vstack([centers, [0.04, 0.04]])
    
    # Initial radii
    r_init = 0.04
    radii = np.full(n, r_init)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Constraints
    cons = [
        {'type': 'ineq', 'fun': bound_con, 'args': (n,)},
        {'type': 'ineq', 'fun': pair_con, 'args': (n,)}
    ]
    
    # Run Optimization
    # SLSQP is suitable for this type of constrained non-linear problem
    res = minimize(obj_func, x0, args=(n,), method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12})
    
    x_opt = res.x
    
    # Extract results
    centers_opt = np.zeros((n, 2))
    radii_opt = np.zeros(n)
    for i in range(n):
        centers_opt[i, 0] = x_opt[3*i]
        centers_opt[i, 1] = x_opt[3*i+1]
        radii_opt[i] = x_opt[3*i+2]
        
    # Ensure non-negative radii
    radii_opt = np.maximum(radii_opt, 0)
    
    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii
