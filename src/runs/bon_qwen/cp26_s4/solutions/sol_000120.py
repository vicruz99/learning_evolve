# sol_000120 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 40ff4175) state=ca84e2b2 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars):
    """
    Computes all inequality constraints for the packing problem.
    Returns an array where every element must be >= 0 for a valid packing.
    """
    n = len(vars) // 3
    cons = []
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    for i in range(n):
        x = vars[3*i]
        y = vars[3*i+1]
        r = vars[3*i+2]
        cons.append(x - r)          # x - r >= 0
        cons.append(1.0 - x - r)    # 1 - x - r >= 0
        cons.append(y - r)          # y - r >= 0
        cons.append(1.0 - y - r)    # 1 - y - r >= 0
        
    # Pairwise non-overlap constraints: dist(i,j) >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = vars[3*i] - vars[3*j]
            dy = vars[3*i+1] - vars[3*j+1]
            d = np.sqrt(dx*dx + dy*dy)
            cons.append(d - vars[3*i+2] - vars[3*j+2])
            
    return np.array(cons)

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def run_packing():
    n = 26
    r_init = 0.090
    
    # 1. Initialize with a hexagonal grid pattern
    centers = np.zeros((n, 2))
    idx = 0
    y = r_init
    while idx < n:
        x = r_init
        # Offset odd rows to create hexagonal packing
        if idx > 0 and (idx // 5) % 2 == 1:
            x += r_init
        while x <= 1.0 - r_init and idx < n:
            centers[idx] = [x, y]
            idx += 1
            x += 2 * r_init
        y += r_init * np.sqrt(3)
        
    # 2. Apply deterministic perturbation to break symmetry
    for i in range(n):
        centers[i, 0] += 0.0005 * (i % 11)
        centers[i, 1] += 0.0005 * (i % 13)
        
    # Ensure initial positions are strictly feasible
    centers = np.clip(centers, r_init + 0.001, 1.0 - r_init - 0.001)
    
    # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(n * 3)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = r_init
        
    # Define bounds for variables
    bounds = [(0, 1) if k % 3 != 2 else (0.01, 0.5) for k in range(n * 3)]
    
    # Setup constraint dictionary
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # 3. Run SLSQP optimization
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons, 
        options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False}
    )
                   
    # 4. Extract and format results
    final_vars = res.x
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    for i in range(n):
        final_centers[i] = [final_vars[3*i], final_vars[3*i+1]]
        final_radii[i] = final_vars[3*i+2]
        
    return final_centers, final_radii, np.sum(final_radii)
