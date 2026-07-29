# sol_000050 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 594c2933) state=7f044002 sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global penalty coefficients
# High values ensure constraints are respected
PENALTY_COEFF_OVERLAP = 50000.0
PENALTY_COEFF_BOUNDARY = 50000.0

def penalty_function(params, n):
    """
    Computes a penalty for constraint violations.
    params: array of shape (3*n,), [x0, y0, r0, x1, y1, r1, ...]
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    penalty = 0.0
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    # Violations: x < r or x > 1-r (and same for y)
    v_x_low = np.maximum(0, r - x)
    v_x_high = np.maximum(0, x + r - 1)
    v_y_low = np.maximum(0, r - y)
    v_y_high = np.maximum(0, y + r - 1)
    
    boundary_pen = np.sum(v_x_low**2) + np.sum(v_x_high**2) + \
                   np.sum(v_y_low**2) + np.sum(v_y_high**2)
    
    penalty += PENALTY_COEFF_BOUNDARY * boundary_pen
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    # Violation if (r_i + r_j)^2 - dist^2 > 0
    
    # Compute distance squared matrix using broadcasting
    diff_x = x[:, np.newaxis] - x[np.newaxis, :]
    diff_y = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = diff_x**2 + diff_y**2
    
    sum_r = r[:, np.newaxis] + r[np.newaxis, :]
    sum_r_sq = sum_r**2
    
    # Mask for upper triangle to avoid double counting and self-distance
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    violation = np.maximum(0, sum_r_sq - dist_sq)
    overlap_pen = np.sum(violation[mask]**2)
    
    penalty += PENALTY_COEFF_OVERLAP * overlap_pen
    
    return penalty

def objective(params, n):
    """
    Objective to minimize: -sum(radii) + penalty
    Minimizing this is equivalent to maximizing sum(radii) subject to constraints.
    """
    r = params[2::3]
    return -np.sum(r) + penalty_function(params, n)

def run_packing():
    n = 26
    best_params = None
    best_score = -np.inf # We track the actual sum of radii
    
    # List of initial center configurations to try
    inits = []
    
    # 1. Dense Grid Initialization
    # Create a 6x6 grid of points within [0.1, 0.9]
    x_pts = np.linspace(0.1, 0.9, 6) 
    y_pts = np.linspace(0.1, 0.9, 6)
    grid = np.array([[x, y] for x in x_pts for y in y_pts])
    # Shuffle and pick 26 points to ensure diversity
    idx = np.random.permutation(len(grid))
    inits.append(grid[idx[:n]])
    
    # 2. Random Initializations
    # Try 5 different random seeds/configurations
    for _ in range(5):
        inits.append(np.random.rand(n, 2) * 0.8 + 0.1)
        
    # 3. Hexagonal Pattern Initialization
    # Hexagonal packing is denser than square grid
    hex_pts = []
    r_est = 0.12 # Estimate for spacing
    y = r_est
    while y + r_est <= 1.0:
        # Determine row index for shifting
        # Row 0: start at r_est
        # Row 1: start at 2*r_est (shifted)
        # Row 2: start at r_est
        row_idx = int(round((y - r_est) / (r_est * np.sqrt(3))))
        if row_idx % 2 == 0:
            x_start = r_est
        else:
            x_start = 2 * r_est
        
        x = x_start
        while x + r_est <= 1.0:
            if len(hex_pts) < n:
                hex_pts.append([x, y])
            x += 2 * r_est
        y += r_est * np.sqrt(3)
    
    # Fill remaining slots if the pattern didn't yield n points
    while len(hex_pts) < n:
        hex_pts.append(list(np.random.rand(2) * 0.8 + 0.1))
    inits.append(np.array(hex_pts))

    # Run optimization for each initialization
    for centers in inits:
        # Start with a small valid radius to ensure initial feasibility
        r_start = 0.05
        params = np.zeros(3 * n)
        params[0::3] = centers[:, 0]
        params[1::3] = centers[:, 1]
        params[2::3] = r_start
        
        # Bounds for variables: x,y in [0,1], r in [0, 0.5]
        bounds = [(0.0, 1.0) for _ in range(2 * n)] + [(0.0, 0.5) for _ in range(n)]
        
        try:
            # Use L-BFGS-B which handles bound constraints efficiently
            res = minimize(objective, params, args=(n,), method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 10000, 'ftol': 1e-12})
            
            # Extract results
            r_res = res.x[2::3]
            sum_r = np.sum(r_res)
            
            # Check validity via penalty function
            # A valid packing should have negligible penalty
            pen = penalty_function(res.x, n)
            
            # If valid and better than current best
            if pen < 1e-5:
                if sum_r > best_score:
                    best_score = sum_r
                    best_params = res.x
        except Exception:
            continue
            
    if best_params is not None:
        centers = np.column_stack((best_params[0::3], best_params[1::3]))
        radii = best_params[2::3]
        return centers, radii, np.sum(radii)
    
    # Fallback solution if optimization fails
    centers = np.random.rand(n, 2) * 0.8 + 0.1
    radii = np.full(n, 0.01)
    return centers, radii, 0.26
