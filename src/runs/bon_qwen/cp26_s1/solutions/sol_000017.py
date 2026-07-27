# sol_000017 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6773994b) state=fa45067d sum of radii=2.618042 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    # Maximize sum of radii -> minimize negative sum
    return -np.sum(vars[2*N_CIRCLES:])

def constraints(vars):
    c = vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    r = vars[2*N_CIRCLES:]
    vals = []
    
    # Boundary constraints: x >= r, x + r <= 1, y >= r, y + r <= 1
    for i in range(N_CIRCLES):
        vals.extend([
            vars[2*i] - vars[2*N_CIRCLES+i],
            1.0 - vars[2*i] - vars[2*N_CIRCLES+i],
            vars[2*i+1] - vars[2*N_CIRCLES+i],
            1.0 - vars[2*i+1] - vars[2*N_CIRCLES+i]
        ])
        
    # Non-overlap constraints: dist >= r_i + r_j
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            vals.append(dist - (r[i] + r[j]))
            
    return np.array(vals)

def run_packing():
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.full(N_CIRCLES, 0.09)
    
    # Initialize on a hexagonal lattice for high initial density
    idx = 0
    row_h = 0.155
    col_w = 0.185
    for r_idx in range(6):
        for c_idx in range(5):
            if idx >= N_CIRCLES: 
                break
            cx = 0.1 + c_idx * col_w + (r_idx % 2) * (col_w / 2)
            cy = 0.1 + r_idx * row_h
            centers[idx] = [cx, cy]
            idx += 1
            
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0, 1) for _ in range(2*N_CIRCLES)] + [(1e-5, 1) for _ in range(N_CIRCLES)]
    
    cons = {'type': 'ineq', 'fun': constraints}
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-13})
                   
    c_opt = res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    r_opt = res.x[2*N_CIRCLES:]
    
    # Tiny safety shrink to guarantee strict feasibility within numerical tolerance
    r_opt *= 0.99999
    c_opt = np.clip(c_opt, r_opt[:, np.newaxis], 1.0 - r_opt[:, np.newaxis])
    
    return c_opt, r_opt, np.sum(r_opt)
