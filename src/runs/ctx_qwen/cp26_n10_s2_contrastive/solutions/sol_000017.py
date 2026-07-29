# sol_000017 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3abe07e0) state=579946e0 sum of radii=2.550508 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars, n):
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    cons = []
    # Boundary constraints
    for i in range(n):
        cons.append(centers[i, 0] - radii[i])
        cons.append(1.0 - centers[i, 0] - radii[i])
        cons.append(centers[i, 1] - radii[i])
        cons.append(1.0 - centers[i, 1] - radii[i])
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            cons.append(dist - radii[i] - radii[j])
    return np.array(cons)

def constraints_26(v):
    return compute_constraints(v, 26)

def objective_26(v):
    return -np.sum(v[52:])

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Hexagonal grid initialization for dense packing
    centers = np.zeros((n, 2))
    idx = 0
    y = 0.12
    while idx < n:
        x = 0.12
        while idx < n and x <= 0.88:
            centers[idx] = [x, y]
            x += 0.23
            idx += 1
        y += 0.20
    while idx < n:
        centers[idx] = np.random.rand(2) * 0.7 + 0.15
        idx += 1
        
    radii = np.full(n, 0.09)
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: centers in [0,1], radii in [0.01, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.01, 0.5)] * n
    
    cons = {'type': 'ineq', 'fun': constraints_26}
    
    # SLSQP optimization
    try:
        res = minimize(objective_26, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 800, 'ftol': 1e-10, 'disp': False})
        if res.success:
            x_opt = res.x
        else:
            x_opt = x0
            # Accept partial improvement if optimization didn't formally succeed
            if objective_26(res.x) < objective_26(x0):
                x_opt = res.x
    except Exception:
        x_opt = x0
        
    centers = x_opt[:2*n].reshape(n, 2)
    radii = x_opt[2*n:]
    
    # Strict feasibility correction to guarantee validation passes
    for _ in range(300):
        changed = False
        for i in range(n):
            # Boundary checks
            for d in range(2):
                if centers[i, d] - radii[i] < -1e-13:
                    radii[i] = centers[i, d] + 1e-14
                    changed = True
                if centers[i, d] + radii[i] > 1.0 + 1e-13:
                    radii[i] = 1.0 - centers[i, d] + 1e-14
                    changed = True
            # Overlap checks
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < radii[i] + radii[j] - 1e-13:
                    shrink = (radii[i] + radii[j] - dist) * 0.5 + 1e-14
                    radii[i] -= shrink
                    radii[j] -= shrink
                    changed = True
        if not changed:
            break
            
    radii = np.maximum(radii, 0.0)
    return centers, radii, float(np.sum(radii))
