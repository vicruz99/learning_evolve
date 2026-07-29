# sol_000131 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5814cb0d) state=fc212bb3 sum of radii=2.340981 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def objective_function(vars, n, weight):
    """
    Objective function to minimize: -sum(radii) + penalty * violations.
    vars: flattened array [x1, y1, r1, x2, y2, r2, ...]
    n: number of circles
    weight: penalty weight
    """
    # Reshape flat vector to (n, 3) array of [x, y, r]
    c = vars.reshape(-1, 3)
    cx = c[:, 0]
    cy = c[:, 1]
    r = c[:, 2]
    
    # We want to maximize sum(r), so we minimize -sum(r)
    val = -np.sum(r)
    
    # Penalty for boundary violations
    # Circles must be inside [0,1]x[0,1]
    # Constraints: r <= x, r <= 1-x, r <= y, r <= 1-y
    # If violated, r - x > 0 etc.
    v1 = np.maximum(0.0, r - cx)
    v2 = np.maximum(0.0, r - (1.0 - cx))
    v3 = np.maximum(0.0, r - cy)
    v4 = np.maximum(0.0, r - (1.0 - cy))
    
    penalty_bound = np.sum(v1**2 + v2**2 + v3**2 + v4**2)
    
    # Penalty for overlap violations
    # Constraint: dist(i,j) >= r_i + r_j
    # Equivalent to: dist^2 >= (r_i + r_j)^2
    # Violation: (r_i + r_j)^2 - dist^2 > 0
    
    # Compute pairwise squared distances using broadcasting
    # cx is (n,), cx[:, None] is (n, 1)
    diff_x = cx[:, None] - cx[None, :]
    diff_y = cy[:, None] - cy[None, :]
    dist_sq = diff_x**2 + diff_y**2
    
    # Sum of radii squared
    sum_r = r[:, None] + r[None, :]
    sum_r_sq = sum_r**2
    
    # Violation amount (positive means overlap)
    violations = sum_r_sq - dist_sq
    
    # Only consider upper triangle (unique pairs i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    violations = violations[mask]
    
    # Penalty is sum of squared positive violations
    p_overlap = np.sum(np.maximum(0.0, violations)**2)
    
    # Total objective
    return val + weight * (penalty_bound + p_overlap)

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal packing pattern
    # This provides a dense, valid starting configuration.
    centers = []
    radii = []
    r_init = 0.08  # Safe initial radius that fits many circles
    y_curr = r_init
    row_idx = 0
    
    # Fill rows in a hexagonal pattern
    while len(centers) < n and y_curr + r_init <= 1.0:
        # Shift odd rows by r_init to create hexagonal offset
        # Even rows start at x = r, Odd rows start at x = 2r
        x_start = r_init if row_idx % 2 == 0 else 2 * r_init
        x_curr = x_start
        
        while len(centers) < n and x_curr + r_init <= 1.0:
            centers.append([x_curr, y_curr])
            radii.append(r_init)
            x_curr += 2 * r_init
        
        # Move to next row with vertical spacing sqrt(3)/2 * diameter = sqrt(3)*r
        y_curr += r_init * math.sqrt(3)
        row_idx += 1
        
    # Convert to numpy array
    centers = np.array(centers)
    
    # If initialization didn't fill n circles (unlikely with r=0.08), pad
    if centers.shape[0] < n:
        pad_count = n - centers.shape[0]
        pad_centers = np.tile([0.5, 0.5], (pad_count, 1))
        centers = np.vstack([centers, pad_centers])
        radii = np.array(radii + [r_init] * pad_count)
    
    # Ensure exact size n
    centers = centers[:n]
    radii = radii[:n]
    
    # Add small random noise to break symmetry and help optimization escape local minima
    np.random.seed(42)
    centers += np.random.normal(0, 0.005, centers.shape)
    
    # Flatten variables for optimizer: [x1, y1, r1, ..., x26, y26, r26]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Define bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    # Penalty weight - needs to be large enough to enforce constraints
    weight = 5000.0
    
    # 2. Optimization
    # Use L-BFGS-B to minimize the penalized objective
    res = minimize(objective_function, x0, method='L-BFGS-B', bounds=bounds, 
                   args=(n, weight),
                   options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    # Extract optimized centers and radii
    c_final = res.x.reshape(-1, 3)
    centers_final = c_final[:, 0:2]
    radii_final = c_final[:, 2]
    
    # 3. Post-processing to ensure strict validity
    
    # Enforce boundary constraints (clamp radii if they touch/exit boundaries)
    for i in range(n):
        x, y = centers_final[i]
        r = radii_final[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r:
            radii_final[i] = max_r
            
    # Resolve overlaps by shrinking radii if any remain (safety net for numerical errors)
    changed = True
    iterations = 0
    while changed and iterations < 100:
        changed = False
        iterations += 1
        for i in range(n):
            for j in range(i+1, n):
                dx = centers_final[i, 0] - centers_final[j, 0]
                dy = centers_final[i, 1] - centers_final[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                required = radii_final[i] + radii_final[j]
                # Check for overlap with tolerance
                if dist < required - 1e-9:
                    if required > 1e-12:
                        scale = dist / required
                        radii_final[i] *= scale
                        radii_final[j] *= scale
                    else:
                        radii_final[i] = 0.0
                        radii_final[j] = 0.0
                    changed = True
                    
    sum_radii = np.sum(radii_final)
    
    return centers_final, radii_final, sum_radii
