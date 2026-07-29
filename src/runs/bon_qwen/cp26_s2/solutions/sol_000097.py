# sol_000097 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2a23e4d6) state=1977702a sum of radii=1.675678 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _compute_objective(vars, n):
    """Compute the penalized objective for the packing optimization."""
    cx = vars[0::3]
    cy = vars[1::3]
    r = vars[2::3]
    
    pen = 0.0
    # Overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx[i] - cx[j]
            dy = cy[i] - cy[j]
            d = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - d
            if overlap > 1e-10:
                pen += overlap ** 2
        
        # Boundary penalty
        pen += max(0.0, r[i] - cx[i]) ** 2
        pen += max(0.0, cx[i] + r[i] - 1.0) ** 2
        pen += max(0.0, r[i] - cy[i]) ** 2
        pen += max(0.0, cy[i] + r[i] - 1.0) ** 2
        
    # Maximize sum of radii <=> minimize negative sum
    return -np.sum(r) + 2000.0 * pen

def run_packing():
    n = 26
    
    # Initialize centers in a hexagonal lattice pattern
    centers = []
    row_cfg = [6, 5, 6, 5, 4]  # 26 circles total
    y = 0.1
    for i, count in enumerate(row_cfg):
        x = 0.1
        if i % 2 == 1:
            x = 0.15  # Hexagonal offset
        for _ in range(count):
            centers.append([x, y])
            x += 0.15
        y += 0.15 * np.sqrt(3)
    centers = np.array(centers)
    radii = np.full(n, 0.085)
    
    # Flatten to optimization variables: [x0, y0, r0, x1, y1, r1, ...]
    vars0 = np.zeros(3 * n)
    for i in range(n):
        vars0[3 * i] = centers[i, 0]
        vars0[3 * i + 1] = centers[i, 1]
        vars0[3 * i + 2] = radii[i]
        
    # Bounds for variables: centers in [0,1], radii in [1e-10, 0.2]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-10, 0.2)] * n
    
    def objective(vars):
        return _compute_objective(vars, n)

    # Run optimization
    res = minimize(objective, vars0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 20000, 'ftol': 1e-15, 'gtol': 1e-10})
                   
    cx_opt = res.x[0::3]
    cy_opt = res.x[1::3]
    r_opt = res.x[2::3]
    
    centers_opt = np.column_stack((cx_opt, cy_opt))
    
    # Post-processing to strictly enforce constraints within validation tolerance
    for _ in range(200):
        changed = False
        for i in range(n):
            # Boundary constraints
            max_r_bound = min(cx_opt[i], 1.0 - cx_opt[i], cy_opt[i], 1.0 - cy_opt[i])
            if r_opt[i] > max_r_bound:
                r_opt[i] = max_r_bound
                changed = True
                
            # Pairwise non-overlap
            for j in range(i + 1, n):
                dx = cx_opt[i] - cx_opt[j]
                dy = cy_opt[i] - cy_opt[j]
                d = np.sqrt(dx * dx + dy * dy)
                max_r_sum = d - 1e-12
                if r_opt[i] + r_opt[j] > max_r_sum:
                    shrink = (r_opt[i] + r_opt[j] - max_r_sum) / 2.0
                    r_opt[i] -= shrink
                    r_opt[j] -= shrink
                    changed = True
                    
        r_opt = np.maximum(r_opt, 1e-12)
        if not changed:
            break
            
    return centers_opt, r_opt, np.sum(r_opt)
