# sol_000300 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bdf692b1) state=ce1ebede sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def obj_func(vars_):
    """Objective: Maximize sum of radii -> Minimize negative sum."""
    return -vars_[-1] * N_CIRCLES

def con_bound(vars_):
    """Boundary constraints: circles must be inside [0, 1]^2."""
    c = vars_[:-1].reshape(-1, 2)
    r = vars_[-1]
    return np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])

def con_overlap(vars_):
    """Overlap constraints: distance between centers >= sum of radii."""
    c = vars_[:-1].reshape(-1, 2)
    r = vars_[-1]
    # Compute pairwise squared distances efficiently
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    # Only check upper triangle to avoid duplicates
    idx = np.triu_indices(N_CIRCLES, k=1)
    return dist2[idx] - (2.0 * r)**2

def run_packing():
    # 1. Initial Configuration: Hexagonal lattice pattern 6-5-6-5-4
    row_counts = [6, 5, 6, 5, 4]
    centers_list = []
    r_est = 0.08
    dx = 2.0 * r_est
    dy = np.sqrt(3.0) * r_est
    
    y_curr = 0.0
    for i, cnt in enumerate(row_counts):
        x_start = 0.0 if i % 2 == 0 else dx / 2.0
        for j in range(cnt):
            centers_list.append([x_start + j * dx, y_curr])
        y_curr += dy
        
    centers = np.array(centers_list)
    
    # Scale and center in [0, 1] to ensure initial feasibility
    min_xy = centers.min(axis=0)
    max_xy = centers.max(axis=0)
    span = max_xy - min_xy
    scale = 0.85 / span.max() 
    centers = (centers - min_xy) * scale + 0.5 * (1.0 - scale * span)
    
    # Initial guess for variables: x1, y1, ..., x26, y26, r
    r_init = 0.07
    x0 = np.concatenate([centers.flatten(), [r_init]])
    
    # Bounds for variables
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)]
    
    # Define constraints
    constraints = [
        {'type': 'ineq', 'fun': con_bound},
        {'type': 'ineq', 'fun': con_overlap}
    ]
    
    # 2. Run Optimization
    res = minimize(obj_func, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 5000, 'ftol': 1e-12})
    
    # 3. Extract results
    if res.success:
        c_opt = res.x[:-1].reshape(-1, 2)
        r_opt = res.x[-1]
        # Ensure non-negative radius
        r_opt = max(0.0, r_opt)
        radii_opt = np.full(N_CIRCLES, r_opt)
        return c_opt, radii_opt, np.sum(radii_opt)
    else:
        # Fallback: Return initial scaled configuration with safe radius
        r_safe = 0.05
        return centers, np.full(N_CIRCLES, r_safe), r_safe * N_CIRCLES
