# sol_000217 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 722eaafb) state=0908f115 sum of radii=2.611162 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective function to minimize (negative sum of radii)"""
    return -np.sum(vars[2::3])

def boundary_constraints(vars):
    """Constraints to keep circles inside the unit square"""
    c = np.empty(4 * N_CIRCLES)
    for i in range(N_CIRCLES):
        x, y, r = vars[3*i], vars[3*i+1], vars[3*i+2]
        idx = 4*i
        c[idx] = x - r
        c[idx+1] = 1 - x - r
        c[idx+2] = y - r
        c[idx+3] = 1 - y - r
    return c

def overlap_constraints(vars):
    """Constraints to prevent circles from overlapping"""
    c = np.empty(N_CIRCLES * (N_CIRCLES - 1) // 2)
    k = 0
    for i in range(N_CIRCLES):
        xi, yi, ri = vars[3*i], vars[3*i+1], vars[3*i+2]
        for j in range(i + 1, N_CIRCLES):
            xj, yj, rj = vars[3*j], vars[3*j+1], vars[3*j+2]
            c[k] = (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            k += 1
    return c

def run_packing():
    np.random.seed(42)
    best_sum = 0.0
    best_vars = None
    
    # Try multiple initial configurations to escape local optima
    for trial in range(5):
        centers = np.zeros((N_CIRCLES, 2))
        radii = np.zeros(N_CIRCLES)
        
        # Initialize with a hexagonal-like pattern
        # 6 rows with counts summing to 26
        row_counts = [4, 5, 4, 5, 4, 4]
        idx = 0
        for r_idx, count in enumerate(row_counts):
            y = 0.12 + r_idx * 0.135
            for c_idx in range(count):
                x = 0.12 + c_idx * 0.15 + (r_idx % 2) * 0.075
                centers[idx] = [x, y]
                idx += 1
                
        # Add slight random perturbation
        centers += np.random.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        radii[:] = 0.05
        
        # Flatten to optimization variables: [x0, y0, r0, x1, y1, r1, ...]
        vars0 = np.zeros(3 * N_CIRCLES)
        for i in range(N_CIRCLES):
            vars0[3*i] = centers[i, 0]
            vars0[3*i+1] = centers[i, 1]
            vars0[3*i+2] = radii[i]
            
        bounds = [(0, 1), (0, 1), (0, 0.5)] * N_CIRCLES
        cons = [
            {'type': 'ineq', 'fun': boundary_constraints},
            {'type': 'ineq', 'fun': overlap_constraints}
        ]
        
        res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_vars = res.x
            
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.zeros(N_CIRCLES)
    for i in range(N_CIRCLES):
        centers[i, 0] = best_vars[3*i]
        centers[i, 1] = best_vars[3*i+1]
        radii[i] = best_vars[3*i+2]
        
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, np.sum(radii)
