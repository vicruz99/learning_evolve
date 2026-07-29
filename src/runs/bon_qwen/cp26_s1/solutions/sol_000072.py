# sol_000072 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3353d097) state=d39e9f4e sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(v):
    """Objective: maximize sum of radii (minimize negative sum)"""
    return -np.sum(v[2::3])

def constraints_func(v):
    """Returns array of inequality constraints >= 0"""
    n = N_CIRCLES
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    cons = [
        c[:, 0] - r,          # x - r >= 0
        1.0 - c[:, 0] - r,    # 1 - x - r >= 0
        c[:, 1] - r,          # y - r >= 0
        1.0 - c[:, 1] - r     # 1 - y - r >= 0
    ]
    
    # Pairwise non-overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
    diff = c[:, None, :] - c[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum_sq = (r[:, None] + r[None, :])**2
    
    idx = np.triu_indices(n, k=1)
    cons.append(dist_sq[idx] - r_sum_sq[idx])
    
    return np.concatenate(cons)

def run_packing():
    n = N_CIRCLES
    r_init = 0.09
    centers = []
    # Hexagonal arrangement counts to get exactly 26 circles
    row_counts = [5, 4, 5, 4, 5, 3]
    
    for row, count in enumerate(row_counts):
        y = r_init + row * r_init * np.sqrt(3)
        for col in range(count):
            x = r_init + col * 2 * r_init + (row % 2) * r_init
            centers.append([x, y])
            
    centers = np.array(centers)
    radii = np.full(n, r_init)
    
    # Flatten initial guess: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 1.0)] * n
    cons_dict = {'type': 'ineq', 'fun': constraints_func}
    
    # Run optimization
    res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons_dict,
                   options={'ftol': 1e-14, 'maxiter': 5000, 'disp': False})
                   
    c_opt = res.x[:2*n].reshape(n, 2)
    r_opt = res.x[2*n:]
    r_opt = np.maximum(r_opt, 1e-9)
    
    # Robust post-processing to guarantee strict feasibility
    alpha = 1.0
    for i in range(n):
        x, y, r = c_opt[i, 0], c_opt[i, 1], r_opt[i]
        if r > 1e-12:
            alpha = min(alpha, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((c_opt[i,0]-c_opt[j,0])**2 + (c_opt[i,1]-c_opt[j,1])**2)
            rs = r_opt[i] + r_opt[j]
            if rs > 1e-12:
                alpha = min(alpha, d / rs)
                
    if alpha < 1.0:
        r_opt *= alpha
        for i in range(n):
            r_opt[i] = min(r_opt[i], c_opt[i,0], 1.0-c_opt[i,0], c_opt[i,1], 1.0-c_opt[i,1])
            
    return c_opt, r_opt, np.sum(r_opt)
