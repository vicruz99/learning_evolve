# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a75b8609) state=d8f8ce11 sum of radii=2.359792 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(vars[2::3])

def get_constraints(n):
    """Generate constraint dictionaries for SLSQP"""
    constraints = []
    
    # Boundary constraints for each circle
    for i in range(n):
        # x >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i] - v[3*i+2]})
        # 1 - x >= r  => 1 >= x + r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i+2]})
        # y >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*i+1] - v[3*i+2]})
        # 1 - y >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[3*i+1] - v[3*i+2]})
        
    # Non-overlap constraints for each pair
    for i in range(n):
        for j in range(i + 1, n):
            # dist^2 >= (r_i + r_j)^2
            constraints.append({
                'type': 'ineq', 
                'fun': lambda v, i=i, j=j: 
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })
            
    return constraints

def run_packing():
    n = 26
    
    # Initial configuration: 5x5 grid + 1 circle in center
    # Start with small radii to ensure feasibility
    centers = np.zeros((n, 2))
    r_init = 0.085
    k = 0
    for i in range(5):
        for j in range(5):
            centers[k] = [0.1 + i * 0.2, 0.1 + j * 0.2]
            k += 1
    centers[25] = [0.5, 0.5]
    
    # Deterministic perturbation to break symmetry and help optimization
    for i in range(n):
        centers[i][0] += 0.002 * (i % 4 - 2)
        centers[i][1] += 0.002 * (i % 6 - 3)
        
    # Flatten to optimization variable vector: [x1, y1, r1, x2, y2, r2, ...]
    vars0 = np.zeros(3 * n)
    for i in range(n):
        vars0[3*i] = centers[i][0]
        vars0[3*i+1] = centers[i][1]
        vars0[3*i+2] = r_init
        
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Generate constraints
    constraints = get_constraints(n)
    
    # Run optimization
    res = minimize(
        objective, 
        vars0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False}
    )
    
    # Extract results
    vars_opt = res.x
    centers_opt = np.array([[vars_opt[3*i], vars_opt[3*i+1]] for i in range(n)])
    radii_opt = vars_opt[2::3]
    
    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, sum_radii
