# sol_000371 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b75b923f) state=55669d60 sum of radii=2.585460 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(z, n):
    """Objective: maximize sum of radii (minimize negative sum)"""
    radii = z[2*n:]
    return -np.sum(radii)

def constraint_func(z, n):
    """Returns array of inequality constraints >= 0"""
    centers = z[:2*n].reshape(n, 2)
    radii = z[2*n:]
    
    con = []
    # Boundary constraints: 0 <= x-r, r <= x <= 1-r, 0 <= y-r, r <= y <= 1-r
    for i in range(n):
        con.append(centers[i, 0] - radii[i])
        con.append(1.0 - centers[i, 0] - radii[i])
        con.append(centers[i, 1] - radii[i])
        con.append(1.0 - centers[i, 1] - radii[i])
        
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            r_sum = radii[i] + radii[j]
            con.append(dist_sq - r_sum*r_sum)
            
    return np.array(con)

def run_packing():
    n = 26
    
    # Structured hexagonal-like initialization for better convergence
    centers = np.zeros((n, 2))
    idx = 0
    # 5 rows with alternating counts to approximate hexagonal density
    row_counts = [6, 5, 6, 5, 4]
    y_step = 1.0 / 6.0
    
    for r_idx, count in enumerate(row_counts):
        y = (r_idx + 1) * y_step
        x_step = 1.0 / (count + 1)
        # Shift alternate rows for hexagonal packing effect
        offset = x_step / 2 if r_idx % 2 == 1 else 0
        
        for c in range(count):
            x = (c + 1) * x_step + offset
            centers[idx] = [np.clip(x, 0.1, 0.9), np.clip(y, 0.1, 0.9)]
            idx += 1
            
    # Start with small feasible radii
    radii = np.full(n, 0.06)
    z0 = np.concatenate([centers.ravel(), radii])
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'args': (n,), 'fun': constraint_func}
    
    best_z = z0
    best_val = -np.sum(radii)
    
    # Run SLSQP optimization
    res = minimize(objective_func, z0, args=(n,), method='SLSQP', bounds=bounds, 
                   constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
                   
    if -res.fun > best_val:
        best_z = res.x
        best_val = -res.fun
        
    centers = best_z[:2*n].reshape(n, 2)
    radii = best_z[2*n:]
    radii = np.maximum(radii, 0.0)
    
    # Post-processing: ensure strict validity within tolerance
    min_ratio = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            r_sum = radii[i] + radii[j]
            if dist < r_sum - 1e-9:
                min_ratio = min(min_ratio, dist / r_sum)
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            min_ratio = 0.999
            
    radii *= min_ratio
    best_val = np.sum(radii)
    
    return centers, radii, best_val
