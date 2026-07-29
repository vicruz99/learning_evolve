# sol_000134 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5da4630c) state=ac9ba782 sum of radii=2.564464 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """
    Objective function: minimize negative sum of radii.
    vars is a 1D array: [x0, y0, r0, x1, y1, r1, ...]
    """
    # Radii are at indices 2, 5, 8, ...
    return -np.sum(vars[2::3])

def all_boundary_constraints_vec(vars):
    """
    Vectorized boundary constraints.
    Returns an array of values that must be >= 0.
    Constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    """
    n = len(vars) // 3
    x = vars[::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])

def all_overlap_constraints_vec(vars):
    """
    Vectorized overlap constraints.
    Returns an array of values that must be >= 0.
    Constraints: dist(i, j)^2 >= (r_i + r_j)^2 for all i < j
    """
    n = len(vars) // 3
    x = vars[::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Compute pairwise squared Euclidean distances
    # dx[i, j] = x[i] - x[j]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    # Compute squared sum of radii
    # sum_r[i, j] = r[i] + r[j]
    sum_r = r[:, None] + r[None, :]
    rad_sq = sum_r**2
    
    # We only need constraints for i < j (upper triangle of the matrix)
    # np.triu_indices(n, k=1) gives indices for the upper triangle excluding diagonal
    idx = np.triu_indices(n, k=1)
    
    # Constraint: dist_sq - rad_sq >= 0
    return dist_sq[idx] - rad_sq[idx]

def run_packing():
    n = 26
    
    # 1. Initialize centers using a hexagonal grid pattern for good initial spacing
    centers = []
    step = 0.2
    y = 0.1
    row_idx = 0
    
    # Generate hex grid points
    while y <= 0.95:
        # Alternate offset for hexagonal packing
        offset = 0.1 if row_idx % 2 == 0 else 0.2
        x = offset
        while x <= 0.95:
            centers.append([x, y])
            x += step
        y += step * (np.sqrt(3)/2)
        row_idx += 1
        
    # Fallback to rectangular grid if hex grid doesn't provide enough points
    if len(centers) < n:
        x_coords = np.linspace(0.1, 0.9, 6)
        y_coords = np.linspace(0.1, 0.9, 5)
        centers = []
        for yy in y_coords:
            for xx in x_coords:
                centers.append([xx, yy])
    
    # Select exactly n points
    centers = centers[:n]
    centers = np.array(centers)
    
    # 2. Initial radii (small value to ensure feasibility)
    radii = np.full(n, 0.01)
    
    # 3. Setup initial vector for optimizer
    # Structure: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # 4. Bounds for variables
    # x, y in [0, 1], r in [0, 0.5] (radius cannot exceed 0.5 in unit square)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))
        
    # 5. Constraints
    constraints = [
        {'type': 'ineq', 'fun': all_boundary_constraints_vec},
        {'type': 'ineq', 'fun': all_overlap_constraints_vec}
    ]
    
    # 6. Run optimization
    # SLSQP is suitable for nonlinear constraints
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
        x_opt = res.x
    except Exception:
        # Fallback in case of optimization failure
        x_opt = x0
        
    # 7. Extract results
    centers_opt = np.zeros((n, 2))
    radii_opt = np.zeros(n)
    for i in range(n):
        centers_opt[i, 0] = x_opt[3*i]
        centers_opt[i, 1] = x_opt[3*i+1]
        radii_opt[i] = x_opt[3*i+2]
        
    # Ensure radii are non-negative (handling potential numerical noise)
    radii_opt = np.maximum(radii_opt, 0.0)
    
    # Calculate sum of radii
    sum_radii = float(np.sum(radii_opt))
    
    # Basic sanity check / cleanup for NaNs
    if np.isnan(centers_opt).any() or np.isnan(radii_opt).any():
        # Fallback to initial valid configuration
        centers_opt = centers
        radii_opt = radii
        sum_radii = float(np.sum(radii_opt))
        
    return centers_opt, radii_opt, sum_radii
