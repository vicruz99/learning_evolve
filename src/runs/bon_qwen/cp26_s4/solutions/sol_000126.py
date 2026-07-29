# sol_000126 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d5ce57f9) state=245b6a3c sum of radii=2.467727 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Assumes equal radii for the optimal sum.
    """
    n = 26
    
    # 1. Initialization: Generate a hexagonal lattice pattern
    # Estimate initial diameter based on hexagonal packing density
    # n * pi * (s/2)^2 <= 0.9069 * Area
    # s <= 2 * sqrt(0.9069 / (n * pi))
    s_est = 2 * np.sqrt(0.9069 / (n * np.pi))
    s_init = s_est * 0.85 # Safety margin for boundary effects
    
    points = []
    h = s_init * np.sqrt(3) / 2
    y = s_init / 2
    row_idx = 0
    
    # Generate points in a hexagonal arrangement
    while y <= 1 - s_init / 2 + 1e-6:
        # Alternate starting x position for staggered rows
        x_start = s_init / 2 if row_idx % 2 == 0 else s_init
        
        x = x_start
        while x <= 1 - s_init / 2 + 1e-6:
            points.append([x, y])
            x += s_init
        
        y += h
        row_idx += 1
        if len(points) >= n:
            break
            
    # Trim or pad if necessary (trimming here as we likely have enough)
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        # Fallback: fill remaining with random points if grid was too sparse
        for _ in range(n - len(points)):
            points.append([0.5, 0.5]) 
            
    init_centers = np.array(points)
    
    # 2. Optimization Setup
    # Variables: [x1, y1, ..., xn, yn, s]
    x0 = np.hstack((init_centers.flatten(), [s_init]))
    
    def objective(vars):
        # Maximize s => minimize -s
        return -vars[2 * n]
    
    def constraints(vars):
        s = vars[2 * n]
        centers = vars[:2 * n].reshape(n, 2)
        cons = []
        
        # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
        # where r = s/2
        r = s / 2.0
        cons.extend(centers[:, 0] - r)
        cons.extend(1 - r - centers[:, 0])
        cons.extend(centers[:, 1] - r)
        cons.extend(1 - r - centers[:, 1])
        
        # Non-overlap constraints: dist(i, j) >= s
        # Using squared distance to avoid sqrt in constraint evaluation
        i, j = np.triu_indices(n, k=1)
        diffs = centers[i] - centers[j]
        dist_sq = np.sum(diffs**2, axis=1)
        cons.extend(dist_sq - s**2)
        
        return np.array(cons)
    
    # Bounds for variables
    # Centers in [0, 1], s in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)]
    
    # Run Optimization
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints={'type': 'ineq', 'fun': constraints},
        options={'maxiter': 2000, 'ftol': 1e-9}
    )
    
    # 3. Extract Results
    optimal_centers = res.x[:2 * n].reshape(n, 2)
    optimal_s = res.x[2 * n]
    optimal_r = optimal_s / 2.0
    
    # Ensure radii are non-negative and valid
    radii = np.full(n, max(0.0, optimal_r))
    
    return optimal_centers, radii, np.sum(radii)
