# sol_000113 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 34ab26db) state=ff5a154e sum of radii=1.083333 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(x):
    return -x[52]

def constraint_bound(x):
    r = x[52]
    c = []
    for i in range(N):
        c.append(x[2*i] - r)
        c.append(1 - x[2*i] - r)
        c.append(x[2*i+1] - r)
        c.append(1 - x[2*i+1] - r)
    return np.array(c)

def constraint_dist(x):
    r = x[52]
    c = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[2*i] - x[2*j]
            dy = x[2*i+1] - x[2*j+1]
            c.append(dx*dx + dy*dy - 4*r*r)
    return np.array(c)

def run_packing():
    # Initial placement: hexagonal-ish layout with 26 circles
    r_init = 0.07
    centers = []
    rows = [6, 4, 6, 4, 6]  # Total 26 circles
    dy = np.sqrt(3) * r_init
    
    for i, cnt in enumerate(rows):
        y = r_init + i * dy
        xs = np.linspace(r_init, 1 - r_init, cnt)
        for x in xs:
            centers.append([x, y])
            
    centers = np.array(centers)
    # Deterministic symmetry breaking
    centers += np.arange(52).reshape(26, 2) * 1e-5
    x0 = np.concatenate([centers.flatten(), [r_init]])
    
    cons = [
        {'type': 'ineq', 'fun': constraint_bound},
        {'type': 'ineq', 'fun': constraint_dist}
    ]
    
    bounds = [(0.0, 1.0)] * 52 + [(0.0, 0.5)]
    
    # Optimize positions and shared radius
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 1500, 'ftol': 1e-12})
                   
    centers_opt = res.x[:52].reshape((N, 2))
    # Ensure centers stay strictly inside bounds
    centers_opt = np.clip(centers_opt, 1e-9, 1 - 1e-9)
    
    # Compute the true maximum feasible radius from optimized centers
    min_dist = 1.0
    for i in range(N):
        xi, yi = centers_opt[i]
        # Distance to boundaries
        min_dist = min(min_dist, xi, 1 - xi, yi, 1 - yi)
        # Distance to other circles
        for j in range(i + 1, N):
            dx = centers_opt[i, 0] - centers_opt[j, 0]
            dy = centers_opt[i, 1] - centers_opt[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < min_dist:
                min_dist = d
                
    r_final = min_dist / 2.0
    radii_opt = np.full(N, r_final)
    
    return centers_opt, radii_opt, N * r_final
