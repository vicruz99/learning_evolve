# sol_000190 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4dd6d242) state=44f68646 sum of radii=2.559881 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(v):
    """Objective function to maximize sum of radii (minimize negative sum)"""
    n = 26
    return -np.sum(v[2*n:])

def constraint_x_min(v, i):
    return v[2*i] - v[2*26 + i]

def constraint_x_max(v, i):
    return 1.0 - v[2*i] - v[2*26 + i]

def constraint_y_min(v, i):
    return v[2*i + 1] - v[2*26 + i]

def constraint_y_max(v, i):
    return 1.0 - v[2*i + 1] - v[2*26 + i]

def constraint_overlap(v, i, j):
    """Non-overlap constraint: squared distance >= squared sum of radii"""
    xi, yi = v[2*i], v[2*i+1]
    xj, yj = v[2*j], v[2*j+1]
    ri, rj = v[2*26 + i], v[2*26 + j]
    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2

def run_packing():
    n = 26
    np.random.seed(42)
    
    # Initial feasible configuration: random positions with small radii
    centers = np.random.rand(n, 2) * 0.7 + 0.15
    radii = np.full(n, 0.02)
    v0 = np.concatenate([centers.flatten(), radii])
    
    # Variable bounds
    bounds = [(0, 1)] * (2 * n) + [(1e-9, 0.5)] * n
    
    # Define constraints
    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': constraint_x_min, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': constraint_x_max, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': constraint_y_min, 'args': (i,)})
        constraints.append({'type': 'ineq', 'fun': constraint_y_max, 'args': (i,)})
        
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_overlap, 'args': (i, j)})
            
    try:
        # Run optimization
        res = minimize(objective, v0, bounds=bounds, constraints=constraints, 
                       method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                       
        if res.success:
            centers_opt = res.x[:2*n].reshape(n, 2)
            radii_opt = res.x[2*n:]
            # Slight shrinkage ensures strict validity against numerical tolerances
            radii_opt = np.maximum(radii_opt * 0.99999, 1e-9)
            return centers_opt, radii_opt, np.sum(radii_opt)
    except Exception:
        pass
        
    # Fallback to initial configuration if optimization fails
    return centers, radii, np.sum(radii)
