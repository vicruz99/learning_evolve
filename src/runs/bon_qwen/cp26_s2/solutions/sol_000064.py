# sol_000064 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4705e2a5) state=e863e63f sum of radii=2.594362 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def objective(vars):
    # vars is a 1D array of shape (3*n,) interleaved as [x1, y1, r1, x2, y2, r2, ...]
    # We want to maximize sum of radii, so we minimize the negative sum
    radii = vars[2::3]
    return -np.sum(radii)

def constraint_fun(vars):
    n = len(vars) // 3
    c = vars.reshape(n, 3)
    n_con = 4 * n + n * (n - 1) // 2
    con = np.zeros(n_con)
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con[:n] = c[:, 0] - c[:, 2]
    con[n:2*n] = 1.0 - c[:, 0] - c[:, 2]
    con[2*n:3*n] = c[:, 1] - c[:, 2]
    con[3*n:4*n] = 1.0 - c[:, 1] - c[:, 2]
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    # Vectorized distance computation
    dx = c[:, np.newaxis, 0] - c[:, 0]
    dy = c[:, np.newaxis, 1] - c[:, 1]
    dist_mat = np.sqrt(dx**2 + dy**2)
    
    r_sum_mat = c[:, np.newaxis, 2] + c[:, 2]
    overlap_mat = dist_mat - r_sum_mat
    
    # Extract upper triangle indices (excluding diagonal)
    idxs = np.triu_indices(n, k=1)
    con[4*n:] = overlap_mat[idxs]
    
    return con

def run_packing():
    n = 26
    best_result = None
    best_obj = np.inf
    
    # Bounds for variables: x, y in [0, 1], r in [0, 1]
    bounds = [(0.0, 1.0) for _ in range(3 * n)]
    
    # Nonlinear constraints: all constraint values must be >= 0
    con = NonlinearConstraint(constraint_fun, lb=0.0, ub=np.inf)
    
    initial_configs = []
    
    # Config 1: Hexagonal lattice pattern (rows of 5, 6, 5, 6, 4)
    cfg1 = np.zeros(3 * n)
    r0 = 0.085
    rows = [5, 6, 5, 6, 4]
    y_pos = r0
    c_idx = 0
    for r_idx, count in enumerate(rows):
        x_start = r0 if r_idx % 2 == 0 else 2 * r0
        x_step = 2 * r0
        for col in range(count):
            if c_idx < n:
                x = x_start + col * x_step
                # Ensure initial positions respect boundaries
                cfg1[3 * c_idx] = np.clip(x, r0, 1.0 - r0)
                cfg1[3 * c_idx + 1] = y_pos
                cfg1[3 * c_idx + 2] = r0
                c_idx += 1
        y_pos += np.sqrt(3) * r0
    initial_configs.append(cfg1)
    
    # Config 2: Structured grid with slight perturbation
    cfg2 = np.zeros(3 * n)
    r0 = 0.08
    idx = 0
    y = r0
    while idx < n:
        x = r0
        while x <= 1.0 - r0 and idx < n:
            cfg2[3 * idx] = x + (0.02 if (idx % 2) else 0.0)
            cfg2[3 * idx + 1] = y + (0.02 if (idx // 5) % 2 else 0.0)
            cfg2[3 * idx + 2] = r0
            idx += 1
            x += 2 * r0
        y += 2 * r0
        if y > 1.0 - r0: break
    initial_configs.append(cfg2)
    
    # Config 3: Random valid placement
    np.random.seed(123)
    cfg3 = np.random.rand(3 * n) * 0.7 + 0.15
    cfg3[2::3] = 0.04 + np.random.rand(n) * 0.02
    # Project radii to satisfy boundary constraints initially
    for i in range(n):
        lim = min(cfg3[3*i], 1-cfg3[3*i], cfg3[3*i+1], 1-cfg3[3*i+1])
        cfg3[3*i+2] = min(cfg3[3*i+2], lim)
    initial_configs.append(cfg3)
    
    for x0 in initial_configs:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=con,
                           options={'maxiter': 3000, 'ftol': 1e-11, 'disp': False})
            if res.fun < best_obj:
                best_obj = res.fun
                best_result = res.x
        except Exception:
            continue
            
    if best_result is None:
        best_result = initial_configs[0]
        
    centers = best_result.reshape(n, 3)[:, :2]
    radii = best_result.reshape(n, 3)[:, 2]
    return centers, radii, float(np.sum(radii))
