# sol_000011 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 92133c71) state=bbbe9bd5 sum of radii=2.594679 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    radii = vars.reshape(N_CIRCLES, 3)[:, 2]
    return -np.sum(radii)

def constraint_func(vars):
    """
    Returns a 1D array of constraint values.
    All constraints are formulated as g(vars) >= 0.
    Includes boundary constraints and pairwise non-overlap constraints.
    """
    pts = vars.reshape(N_CIRCLES, 3)
    centers = pts[:, :2]
    radii = pts[:, 2]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_boundary = np.column_stack([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ]).flatten()
    
    # Non-overlap constraints: dist_sq - (r_i + r_j)^2 >= 0
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
    dist_sq = dx**2 + dy**2
    rad_sum_sq = (radii[:, np.newaxis] + radii[np.newaxis, :])**2
    
    # Extract upper triangular indices for i < j
    i, j = np.triu_indices(N_CIRCLES, k=1)
    c_overlap = dist_sq[i, j] - rad_sum_sq[i, j]
    
    return np.concatenate([c_boundary, c_overlap])

def run_packing():
    best_vars = None
    best_sum = -1.0
    np.random.seed(123)
    
    # Define bounds: x,y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Multiple restarts with perturbed hexagonal initial guesses
    for attempt in range(4):
        vars_init = np.zeros(3 * N_CIRCLES)
        idx = 0
        r_init = 0.04 + np.random.rand() * 0.02
        
        # Hexagonal layout: 6, 5, 6, 5, 4 rows
        rows = [6, 5, 6, 5, 4]
        y_curr = r_init
        for k, count in enumerate(rows):
            x_start = r_init if k % 2 == 0 else 2 * r_init
            for m in range(count):
                vars_init[idx] = x_start + m * 2 * r_init
                vars_init[idx+1] = y_curr
                vars_init[idx+2] = r_init
                idx += 3
            y_curr += r_init * np.sqrt(3)
            
        # Add small random noise to break symmetry and ensure strict feasibility
        vars_init += np.random.randn(len(vars_init)) * 0.005
        
        # Clip to valid bounds
        for i in range(0, len(vars_init), 3):
            vars_init[i] = np.clip(vars_init[i], 0.01, 0.99)
            vars_init[i+1] = np.clip(vars_init[i+1], 0.01, 0.99)
            vars_init[i+2] = np.clip(vars_init[i+2], 0.01, 0.4)
            
        # Local optimization
        res = minimize(objective, vars_init, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'iprint': -1})
        
        curr_sum = -res.fun
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_vars = res.x
            
    # High-precision refinement on the best configuration found
    if best_vars is not None:
        res_final = minimize(objective, best_vars, method='SLSQP', bounds=bounds, 
                             constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'iprint': -1})
        best_vars = res_final.x
        
    centers = best_vars.reshape(N_CIRCLES, 3)[:, :2]
    radii = best_vars.reshape(N_CIRCLES, 3)[:, 2]
    
    # Ensure non-negative radii (safety against numerical drift)
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, np.sum(radii)
