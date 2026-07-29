# sol_000249 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5bb01f44) state=89da6973 sum of radii=2.579619 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

NUM_CIRCLES = 26

def objective(vars):
    """Objective: minimize negative sum of radii"""
    return -np.sum(vars[-NUM_CIRCLES:])

def constr(vars):
    """Constraints: boundary containment and circle non-overlap"""
    c = vars[:2*NUM_CIRCLES].reshape(NUM_CIRCLES, 2)
    r = vars[-NUM_CIRCLES:]
    vals = []
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0, r >= 0
    vals.extend(c[:, 0] - r)
    vals.extend(1.0 - c[:, 0] - r)
    vals.extend(c[:, 1] - r)
    vals.extend(1.0 - c[:, 1] - r)
    vals.extend(r)
    
    # Inter-circle constraints: dist^2 >= (r_i + r_j)^2
    for i in range(NUM_CIRCLES):
        for j in range(i + 1, NUM_CIRCLES):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            dist2 = dx*dx + dy*dy
            vals.append(dist2 - (r[i] + r[j])**2)
            
    return np.array(vals)

def run_packing():
    # Initialize centers in a dense hexagonal-like grid pattern
    centers = np.zeros((NUM_CIRCLES, 2))
    idx = 0
    for row in range(5):
        y = 0.1 + row * 0.2
        x_start = 0.15 if row % 2 == 1 else 0.1
        for col in range(7):
            if idx >= NUM_CIRCLES:
                break
            x = x_start + col * 0.1
            if x <= 0.9:
                centers[idx] = [x, y]
                idx += 1
                
    # Fill any remaining slots if the pattern didn't cover all 26
    for i in range(idx, NUM_CIRCLES):
        centers[i] = [0.2 + (i - idx) * 0.05, 0.5]
        
    # Initial radii small enough to be strictly feasible with the grid
    x0 = np.concatenate([centers.ravel(), np.full(NUM_CIRCLES, 0.03)])
    
    # Variable bounds: centers in [0,1], radii in [0, 0.5]
    bnds = [(0.0, 1.0) for _ in range(2*NUM_CIRCLES)] + [(0.0, 0.5) for _ in range(NUM_CIRCLES)]
    cons = {'type': 'ineq', 'fun': constr}
    
    # Run SLSQP optimizer
    res = minimize(objective, x0, method='SLSQP', constraints=cons, bounds=bnds,
                   options={'maxiter': 1500, 'ftol': 1e-12})
                   
    final_centers = res.x[:2*NUM_CIRCLES].reshape(NUM_CIRCLES, 2)
    final_radii = np.maximum(res.x[-NUM_CIRCLES:], 0.0)
    
    # Return packed configuration
    return final_centers, final_radii, np.sum(final_radii)
