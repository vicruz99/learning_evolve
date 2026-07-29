# sol_000007 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5f9158d7) state=33c0c451 sum of radii=2.602731 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
I_UPPER, J_UPPER = np.triu_indices(N_CIRCLES, k=1)

def objective(x):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Inequality constraints:
    - Pairwise distance >= sum of radii
    - Circle boundaries within [0,1]x[0,1]
    Returns array of constraint values (must be >= 0)
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Distance constraints between all pairs
    dx = cx[I_UPPER] - cx[J_UPPER]
    dy = cy[I_UPPER] - cy[J_UPPER]
    dists = np.hypot(dx, dy)
    c_dist = dists - (r[I_UPPER] + r[J_UPPER])
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_bound = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    return np.concatenate([c_dist, c_bound])

def run_packing():
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    best_sum = 0.0
    best_params = None
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Multiple restarts to escape local optima
    for _ in range(50):
        # Initialize centers randomly in safe inner region, small radii for feasibility
        cx = np.random.uniform(0.15, 0.85, N_CIRCLES)
        cy = np.random.uniform(0.15, 0.85, N_CIRCLES)
        r = np.full(N_CIRCLES, 0.03)
        x0 = np.concatenate([cx, cy, r])
        
        res = minimize(
            objective, 
            x0, 
            method='SLSQP', 
            bounds=bounds,
            constraints=cons, 
            options={'maxiter': 800, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.success:
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_params = res.x.copy()
                
    if best_params is not None:
        centers = np.column_stack((best_params[0::3], best_params[1::3]))
        radii = np.maximum(best_params[2::3], 0.0)  # Ensure non-negative
        return centers, radii, np.sum(radii)
    else:
        # Fallback configuration (should not be reached)
        centers = np.random.rand(N_CIRCLES, 2) * 0.6 + 0.2
        radii = np.full(N_CIRCLES, 0.02)
        return centers, radii, np.sum(radii)
