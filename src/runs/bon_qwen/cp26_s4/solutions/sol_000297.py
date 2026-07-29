# sol_000297 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 57e53fd2) state=e6be2f1b sum of radii=2.157339 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(vars, n, mu):
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:3*n]
    
    # Boundary violations
    v_x1 = np.maximum(0, r - x)
    v_x2 = np.maximum(0, x + r - 1.0)
    v_y1 = np.maximum(0, r - y)
    v_y2 = np.maximum(0, y + r - 1.0)
    penalty_boundary = np.sum(v_x1**2 + v_x2**2 + v_y1**2 + v_y2**2)
    
    # Overlap violations
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    rs = r[:, None] + r[None, :]
    v_overlap = np.maximum(0, rs - dist)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    penalty_overlap = np.sum(v_overlap[mask]**2)
    
    total_penalty = penalty_boundary + penalty_overlap
    return -np.sum(r) + mu * total_penalty

def get_init_config(n, seed):
    np.random.seed(seed)
    xs = []
    ys = []
    rows = 5
    cols = 6
    for i in range(rows):
        for j in range(cols):
            x = 0.15 + j * 0.14 + (i % 2) * 0.07
            y = 0.15 + i * 0.14 * np.sqrt(3) / 2.0
            xs.append(x)
            ys.append(y)
    xs = np.array(xs[:n]) + np.random.uniform(-0.04, 0.04, n)
    ys = np.array(ys[:n]) + np.random.uniform(-0.04, 0.04, n)
    xs = np.clip(xs, 0.05, 0.95)
    ys = np.clip(ys, 0.05, 0.95)
    r = np.full(n, 0.06)
    return np.concatenate([xs, ys, r])

def refine_solution(vars, n, mu, bounds, maxiter=5000):
    res = minimize(compute_loss, vars, args=(n, mu), method='L-BFGS-B', bounds=bounds, options={'maxiter': maxiter, 'ftol': 1e-15})
    return res.x

def ensure_feasibility(centers, radii, n):
    x = centers[:, 0]
    y = centers[:, 1]
    r = radii.copy()
    
    # Adjust for boundaries
    for i in range(n):
        r[i] = min(r[i], x[i], 1.0 - x[i], y[i], 1.0 - y[i])
        
    # Adjust for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx*dx + dy*dy)
            if d < r[i] + r[j]:
                shrink = (r[i] + r[j] - d) / 2.0 + 1e-10
                r[i] -= shrink
                r[j] -= shrink
    return r

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    
    best_vars = None
    best_sum_r = -np.inf
    
    # Multi-restart optimization
    for seed in range(12):
        x0 = get_init_config(n, seed)
        # Stage 1: Moderate penalty to navigate landscape
        vars1 = refine_solution(x0, n, 5000.0, bounds, maxiter=3000)
        # Stage 2: High penalty to strictly enforce constraints and maximize radii
        vars2 = refine_solution(vars1, n, 20000.0, bounds, maxiter=5000)
        
        curr_r = vars2[2*n:3*n]
        curr_sum = np.sum(curr_r)
        if curr_sum > best_sum_r:
            best_sum_r = curr_sum
            best_vars = vars2.copy()
            
    x = best_vars[:n]
    y = best_vars[n:2*n]
    r = best_vars[2*n:3*n]
    
    centers = np.stack((x, y), axis=1)
    r = ensure_feasibility(centers, r, n)
    
    final_sum = np.sum(r)
    return centers, r, final_sum
