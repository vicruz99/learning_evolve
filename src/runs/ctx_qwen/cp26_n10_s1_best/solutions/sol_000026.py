# sol_000026 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 764eb384) state=ca639ee5 sum of radii=2.368571 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_):
    n = 26
    cs = vars_[:2*n].reshape(n, 2)
    rs = vars_[2*n:]
    
    # Objective: minimize negative sum of radii
    obj = -np.sum(rs)
    
    # Penalty for constraint violations
    pen = 0.0
    
    # Boundary constraints: r <= x <= 1-r and r <= y <= 1-r
    pen += np.sum(np.maximum(0.0, rs - cs[:, 0])**2)
    pen += np.sum(np.maximum(0.0, cs[:, 0] + rs - 1.0)**2)
    pen += np.sum(np.maximum(0.0, rs - cs[:, 1])**2)
    pen += np.sum(np.maximum(0.0, cs[:, 1] + rs - 1.0)**2)
    
    # Overlap constraints: dist >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((cs[i] - cs[j])**2) + 1e-16)
            pen += np.maximum(0.0, rs[i] + rs[j] - dist)**2
            
    return obj + 1e6 * pen

def run_packing():
    n = 26
    
    # 1. Initialize with a dense hexagonal pattern
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09)
    
    idx = 0
    y = 0.1
    shift = 0.0
    while idx < n:
        x = 0.1
        while x + 0.09 <= 0.91 and idx < n:
            centers[idx] = [x + shift, y]
            idx += 1
            x += 0.18
        shift = 0.09 if shift == 0.0 else 0.0
        y += 0.09 * np.sqrt(3)
        
    x0 = np.hstack([centers.flatten(), radii])
    
    # 2. Optimization
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.02, 0.5)] * n
    res = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', 
                   options={'maxiter': 30000, 'ftol': 1e-13, 'disp': False})
    
    final_centers = res.x[:2*n].reshape(n, 2)
    final_radii = res.x[2*n:]
    
    # 3. Post-processing to guarantee strict validity within tolerance
    for _ in range(200):
        fixed = True
        
        # Enforce boundary constraints
        for i in range(n):
            x, y = final_centers[i]
            r = final_radii[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r + 1e-12:
                final_radii[i] = max(max_r, 0.0)
                fixed = False
                
        # Enforce non-overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
                req = final_radii[i] + final_radii[j]
                if dist < req - 1e-12:
                    scale = (dist - 1e-12) / req
                    final_radii[i] *= scale
                    final_radii[j] *= scale
                    fixed = False
                    
        if fixed:
            break
            
    return final_centers, final_radii, np.sum(final_radii)
