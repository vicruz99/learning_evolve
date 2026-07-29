# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 79449191) state=2a200448 sum of radii=2.371937 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def get_optimal_radii_sum(centers):
    """
    Given fixed centers, solve the LP to find radii that maximize sum of radii
    subject to non-overlap and boundary constraints.
    
    Args:
        centers: np.array of shape (n, 2)
        
    Returns:
        (sum_radii, radii_array)
    """
    n = centers.shape[0]
    
    # Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
    c_obj = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Precompute distances to avoid repeated sqrt calls in loops if needed,
    # but for n=26 simple loops are fine.
    
    for i in range(n):
        xi, yi = centers[i]
        
        # Boundary constraints: r_i <= dist_to_wall
        # r_i <= xi
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(xi)
        
        # r_i <= 1 - xi
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - xi)
        
        # r_i <= yi
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(yi)
        
        # r_i <= 1 - yi
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(1.0 - yi)
        
        # Pairwise constraints: r_i + r_j <= dist(i, j)
        # Only need j > i to avoid duplicates and self
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
        else:
            # Fallback if LP fails, though unlikely with valid centers
            return 0.0, np.zeros(n)
    except Exception:
        return 0.0, np.zeros(n)

def objective_function(centers_flat):
    """
    Objective function for the center optimizer.
    Returns negative sum of optimal radii.
    """
    n = len(centers_flat) // 2
    centers = centers_flat.reshape((n, 2))
    # Ensure centers are within bounds for validity, though LP handles boundaries for radii.
    # If centers are outside, radii might be negative in LP logic if not careful, 
    # but bounds r>=0 handles it. However, logically centers should be in [0,1].
    # The optimizer bounds will keep them there.
    
    sum_r, radii = get_optimal_radii_sum(centers)
    return -sum_r

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal Grid
    # Approximate radius for hex packing in unit square for 26 circles
    # 26 circles area ~ 0.8 * 1.0 (density) -> 26 * pi * r^2 = 0.8 -> r ~ 0.1
    # Hex spacing: horizontal 2r, vertical r*sqrt(3)
    r_init = 0.11
    dx = 2 * r_init
    dy = r_init * np.sqrt(3)
    
    init_centers = []
    y = r_init
    row_idx = 0
    count = 0
    
    while count < n:
        offset = (row_idx % 2) * (dx / 2)
        x = r_init + offset
        
        while x + r_init <= 1.0 + 1e-9 and count < n:
            init_centers.append([x, y])
            x += dx
            count += 1
        y += dy
        row_idx += 1
        
    # If we somehow didn't fill 26 (unlikely), pad
    while len(init_centers) < n:
        init_centers.append([0.5, 0.5])
        
    init_centers = np.array(init_centers[:n])
    x0 = init_centers.flatten()
    
    # 2. Optimization
    # Bounds for centers: [0, 1]
    bounds_opt = [(0.0, 1.0) for _ in range(2 * n)]
    
    # Use Nelder-Mead for local optimization from good start
    # Max iterations can be high
    res = minimize(objective_function, x0, method='Nelder-Mead', 
                   bounds=bounds_opt, 
                   options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
    
    best_centers = res.x.reshape((n, 2))
    _, best_radii = get_optimal_radii_sum(best_centers)
    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, sum_radii
