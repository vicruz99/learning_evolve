# sol_000032 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8e46300b) state=68d3ef97 sum of radii=2.589314 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def objective(x):
    """
    Objective function to minimize: -sum(radii)
    x is a flattened array [x1, y1, r1, x2, y2, r2, ...]
    """
    n = len(x) // 3
    radii = x.reshape(n, 3)[:, 2]
    return -np.sum(radii)

def constraints_func(x):
    """
    Returns a vector of constraint values.
    All values must be >= 0 for the point to be feasible.
    Constraints:
    1. x_i >= r_i
    2. x_i <= 1 - r_i
    3. y_i >= r_i
    4. y_i <= 1 - r_i
    5. dist(i, j)^2 >= (r_i + r_j)^2 for all i < j
    """
    n = len(x) // 3
    data = x.reshape(n, 3)
    centers = data[:, :2]
    radii = data[:, 2]
    
    constraints = []
    
    # Boundary constraints
    # 1. x >= r
    constraints.append(centers[:, 0] - radii)
    # 2. x <= 1 - r  => 1 - x - r >= 0
    constraints.append(1.0 - centers[:, 0] - radii)
    # 3. y >= r
    constraints.append(centers[:, 1] - radii)
    # 4. y <= 1 - r => 1 - y - r >= 0
    constraints.append(1.0 - centers[:, 1] - radii)
    
    # Overlap constraints
    # Compute pairwise squared distances
    x_coords = centers[:, 0]
    y_coords = centers[:, 1]
    
    # Difference matrices (n, n)
    dx = x_coords[:, np.newaxis] - x_coords[np.newaxis, :]
    dy = y_coords[:, np.newaxis] - y_coords[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Radii sum squared
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    # Constraint value: dist_sq - (r_i + r_j)^2 >= 0
    val = dist_sq - r_sum_sq
    
    # We only need constraints for i < j (upper triangle excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    overlap_constraints = val[mask]
    
    constraints.append(overlap_constraints)
    
    # Concatenate all constraints into a single 1D array
    return np.concatenate(constraints)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    vars_per_circle = 3 # x, y, r
    total_vars = n_circles * vars_per_circle
    
    # --- Initialization ---
    # Start with a hexagonal lattice pattern which is dense.
    # We estimate a radius that fits roughly 26 circles.
    # A safe starting radius to avoid initial overlaps is 0.05, but we place centers
    # based on a larger temp radius to get good spacing.
    r_temp = 0.08 
    dx = 2 * r_temp
    dy = math.sqrt(3) * r_temp
    
    raw_centers = []
    # Generate a 6x5 grid of points (30 points) in hexagonal arrangement
    # We will select the first 26.
    for j in range(5): # 5 rows
        for i in range(6): # 6 columns
            x = i * dx + (j % 2) * (dx / 2)
            y = j * dy
            raw_centers.append([x, y])
            
    raw_centers = np.array(raw_centers)
    
    # Center and scale the pattern to fit within [0.1, 0.9]
    # Leaving margin for radius growth
    min_c = raw_centers.min(axis=0)
    max_c = raw_centers.max(axis=0)
    center_c = (min_c + max_c) / 2
    size_c = max_c - min_c
    scale = 0.8 / np.max(size_c)
    raw_centers = (raw_centers - center_c) * scale + 0.5
    
    selected_centers = raw_centers[:n_circles]
    
    # Initial variables vector
    x0 = np.zeros(total_vars)
    for i in range(n_circles):
        x0[3*i] = selected_centers[i, 0]
        x0[3*i+1] = selected_centers[i, 1]
        x0[3*i+2] = 0.05 # Small initial radius to ensure feasibility
        
    # Bounds for variables
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0))   # x
        bounds.append((0.0, 1.0))   # y
        bounds.append((0.0, 0.5))   # r
        
    # Define constraints
    cons = {'type': 'ineq', 'fun': constraints_func}
    
    # Run optimization
    # SLSQP method supports bounds and non-linear constraints
    try:
        result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, 
                              constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
        x_opt = result.x
    except Exception:
        x_opt = x0

    # Extract results
    centers_final = np.zeros((n_circles, 2))
    radii_final = np.zeros(n_circles)
    
    for i in range(n_circles):
        centers_final[i, 0] = x_opt[3*i]
        centers_final[i, 1] = x_opt[3*i+1]
        radii_final[i] = x_opt[3*i+2]
        
    # Ensure non-negative radii (numerical safety)
    radii_final = np.maximum(radii_final, 0.0)
    
    sum_radii = np.sum(radii_final)
    
    return centers_final, radii_final, sum_radii
