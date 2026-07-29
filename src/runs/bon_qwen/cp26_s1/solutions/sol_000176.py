# sol_000176 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ae68a5b3) state=0e0f6e7b sum of radii=2.617835 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x):
    return -np.sum(x[2::3])

def constraint_fun(x):
    n = len(x) // 3
    num_constraints = 4 * n + n * (n - 1) // 2
    c = np.empty(num_constraints)
    idx = 0
    for i in range(n):
        c[idx] = x[3*i] - x[3*i+2]
        c[idx+1] = 1.0 - x[3*i] - x[3*i+2]
        c[idx+2] = x[3*i+1] - x[3*i+2]
        c[idx+3] = 1.0 - x[3*i+1] - x[3*i+2]
        idx += 4
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[3*i] - x[3*j]
            dy = x[3*i+1] - x[3*j+1]
            dr = x[3*i+2] + x[3*j+2]
            c[idx] = dx**2 + dy**2 - dr**2
            idx += 1
    return c

def run_packing():
    n = 26
    pts = []
    for i in range(6):
        for j in range(5):
            x = (j + 0.5) / 5.0 + (0.5 / 5.0 if i % 2 else 0)
            y = (i + 0.5) / 6.0
            pts.append([x, y])
    pts = np.array(pts[:n])
    
    np.random.seed(42)
    pts += np.random.uniform(-0.02, 0.02, pts.shape)
    r0 = 0.08 + np.random.uniform(0, 0.02, n)
    
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = pts[i, 0]
        x0[3*i+1] = pts[i, 1]
        x0[3*i+2] = r0[i]
        
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 3000, 'ftol': 1e-12})
                   
    centers = np.column_stack((res.x[0::3], res.x[1::3]))
    radii = res.x[2::3]
    
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        radii[i] = max(0.0, radii[i])
        
    return centers, radii, np.sum(radii)
