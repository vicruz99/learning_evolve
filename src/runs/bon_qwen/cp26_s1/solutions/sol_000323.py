# sol_000323 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fe3e1745) state=79096919 sum of radii=2.608631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

# Global constant for the number of circles
N_CIRCLES = 26

def compute_constraints(v):
    """
    Computes all inequality constraints for the packing problem.
    Returns an array of values that must be >= 0.
    Constraints include boundary checks and non-overlap checks.
    
    Args:
        v: 1D numpy array of shape (3*N_CIRCLES,) containing [x0, y0, r0, x1, y1, r1, ...]
        
    Returns:
        numpy array of constraint values.
    """
    constraints_list = []
    
    # Boundary constraints: 4 per circle
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    for i in range(N_CIRCLES):
        idx = 3 * i
        x = v[idx]
        y = v[idx+1]
        r = v[idx+2]
        
        constraints_list.append(x - r)
        constraints_list.append(1.0 - x - r)
        constraints_list.append(y - r)
        constraints_list.append(1.0 - y - r)
        
    # Overlap constraints: (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
    # This ensures distance between centers >= sum of radii
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            idx_i = 3 * i
            idx_j = 3 * j
            
            dx = v[idx_i] - v[idx_j]
            dy = v[idx_i+1] - v[idx_j+1]
            dr = v[idx_i+2] + v[idx_j+2]
            
            val = dx*dx + dy*dy - dr*dr
            constraints_list.append(val)
            
    return np.array(constraints_list)

def objective_func(v):
    """
    Objective: Maximize sum of radii.
    We minimize the negative sum.
    
    Args:
        v: 1D numpy array of variables.
        
    Returns:
        Negative sum of radii.
    """
    # Radii are at indices 2, 5, 8, ...
    radii = v[2::3]
    return -np.sum(radii)

def run_packing():
    """
    Runs the optimization to pack 26 circles in a unit square maximizing sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = N_CIRCLES
    
    # --- 1. Initialization ---
    # Generate a hexagonal grid pattern.
    # We choose row counts that sum to 26 and have an aspect ratio close to 1 (square).
    # 5, 4, 5, 4, 5, 3 sums to 26.
    # Width of rows with 5 items is approx 4.0, rows with 4 items is 3.0.
    # Height grows by sqrt(3)/2 per row.
    row_counts = [5, 4, 5, 4, 5, 3]
    
    points = []
    y_curr = 0
    # Hexagonal grid: x step 1.0, y step sqrt(3)/2
    # Odd rows are offset by 0.5 to nest in gaps
    for r_idx, count in enumerate(row_counts):
        x_offset = 0.5 if r_idx % 2 == 1 else 0.0
        for c_idx in range(count):
            x = c_idx * 1.0 + x_offset
            y = y_curr
            points.append([x, y])
        y_curr += math.sqrt(3) / 2.0
        
    pts_array = np.array(points)
    
    # Center and Scale the points to fit in the unit square [0,1]x[0,1]
    min_x, min_y = pts_array.min(axis=0)
    max_x, max_y = pts_array.max(axis=0)
    w = max_x - min_x
    h = max_y - min_y
    
    # Center the grid at (0.5, 0.5)
    pts_array -= np.array([min_x + w/2, min_y + h/2])
    
    # Scale to fit within [0.05, 0.95] roughly to allow room for initial radii
    scale = 0.45 / max(w, h)
    pts_array *= scale
    pts_array += 0.5
    
    # Initial radii: small enough to ensure no overlap initially
    r_init = 0.02
    radii = np.full(n, r_init)
    
    # Flatten to optimization vector: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = pts_array[i, 0]
        x0[3*i+1] = pts_array[i, 1]
        x0[3*i+2] = r_init
        
    # Bounds for variables: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    # Define constraints
    cons = {'type': 'ineq', 'fun': compute_constraints}
        
    # --- 2. Optimization ---
    try:
        # Use SLSQP solver for non-linear constrained optimization
        res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
        
        # Extract result
        best_x = res.x
        centers = np.zeros((n, 2))
        radii_out = np.zeros(n)
        for i in range(n):
            centers[i, 0] = best_x[3*i]
            centers[i, 1] = best_x[3*i+1]
            radii_out[i] = best_x[3*i+2]
            
        # Ensure radii are non-negative (though bounds should enforce this)
        radii_out = np.maximum(radii_out, 0.0)
        
        sum_r = np.sum(radii_out)
        return centers, radii_out, sum_r
        
    except Exception:
        # Fallback to initial guess if optimization fails
        centers_out = pts_array
        radii_out = np.full(n, r_init)
        return centers_out, radii_out, np.sum(radii_out)
