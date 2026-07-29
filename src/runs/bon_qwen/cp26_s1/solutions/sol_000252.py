# sol_000252 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2f3e441d) state=4176923f sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(x):
    """Objective: minimize negative sum of radii"""
    return -np.sum(x[2*N:])

def constraints(x):
    """Inequality constraints: walls and non-overlap"""
    c = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    # Wall constraints (5 per circle: left, right, bottom, top, non-negative radius)
    w = np.empty(5 * N)
    w[0::5] = c[:, 0] - r
    w[1::5] = 1.0 - c[:, 0] - r
    w[2::5] = c[:, 1] - r
    w[3::5] = 1.0 - c[:, 1] - r
    w[4::5] = r
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    m = N * (N - 1) // 2
    o = np.empty(m)
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            o[idx] = np.sum((c[i] - c[j])**2) - (r[i] + r[j])**2
            idx += 1
            
    return np.concatenate([w, o])

def run_packing():
    np.random.seed(42)
    
    # 1. Initial hexagonal packing configuration
    r_init = 0.09
    centers = []
    y = r_init
    row = 0
    while y < 1.0 - r_init and len(centers) < N:
        x = r_init
        if row % 2 == 1:
            x += r_init  # Hexagonal offset
        while x < 1.0 - r_init and len(centers) < N:
            centers.append([x, y])
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row += 1
        
    centers = np.array(centers[:N])
    radii = np.full(N, 0.06)
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_val = -np.inf
    best_x = x0
    
    # 2. Multiple restarts to avoid local optima
    for _ in range(5):
        x0_trial = x0 + np.random.randn(len(x0)) * 0.002
        x0_trial = np.clip(x0_trial, 0.01, 0.99)
        x0_trial[2*N:] = np.clip(x0_trial[2*N:], 1e-5, 0.4)
        
        try:
            res = minimize(objective, x0_trial, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13})
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x
        except Exception:
            continue
            
    centers_out = best_x[:2*N].reshape(N, 2)
    radii_out = best_x[2*N:]
    
    # 3. Safety adjustment to guarantee validity within numerical tolerance
    for _ in range(30):
        valid = True
        for i in range(N):
            x, y = centers_out[i]
            r = radii_out[i]
            if x - r < -1e-9 or x + r > 1.0 + 1e-9 or y - r < -1e-9 or y + r > 1.0 + 1e-9:
                valid = False
                break
        if valid:
            break
        radii_out *= 0.998  # Tiny uniform shrink if needed
        
    sum_r = float(np.sum(radii_out))
    return centers_out, radii_out, sum_r
