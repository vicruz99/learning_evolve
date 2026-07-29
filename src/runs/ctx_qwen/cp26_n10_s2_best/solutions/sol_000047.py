# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000016 (state 585439f0) state=ac92a3c3 sum of radii=2.624758 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_objective(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2*N_CIRCLES:])

def compute_constraints(vars):
    """Inequality constraints: boundaries and non-overlap."""
    n = N_CIRCLES
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints: dist >= r_i + r_j
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Only need upper triangle pairs to avoid redundancy
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c = np.concatenate([c, (dist - r_sum)[mask]])
    return c

def run_packing():
    best_sum = -1.0
    best_v = None
    bounds = [(0.0, 1.0)] * (2*N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    
    configs = []
    
    # Generate diverse initial configurations from perturbed hexagonal lattices
    for seed in range(35):
        np.random.seed(seed)
        r_init = 0.08 + np.random.uniform(0, 0.04)
        pts = []
        y = r_init
        row = 0
        # Generate enough points to fill N_CIRCLES
        while len(pts) < N_CIRCLES + 10:
            x = r_init if row % 2 == 0 else 2 * r_init
            while x <= 1.0 - r_init and len(pts) < N_CIRCLES + 10:
                pts.append([x, y])
                x += 2 * r_init
            y += r_init * np.sqrt(3)
            row += 1
            
        pts = np.array(pts[:N_CIRCLES])
        # Perturb to break symmetry and explore different local minima
        pts += np.random.uniform(-0.06, 0.06, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        
        # Initial radii slightly randomized to avoid identical starting points
        r_vec = np.full(N_CIRCLES, 0.04) + np.random.uniform(-0.01, 0.01, N_CIRCLES)
        r_vec = np.clip(r_vec, 0.01, 0.1)
        
        configs.append(np.concatenate([pts[:, 0], pts[:, 1], r_vec]))
        
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    for v0 in configs:
        res = minimize(compute_objective, v0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
                       
        # Check feasibility with tolerance
        if np.all(compute_constraints(res.x) >= -1e-9):
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_v = res.x.copy()
                
    # Fallback if optimization fails unexpectedly
    if best_v is None:
        best_v = configs[0]
        
    x = best_v[:N_CIRCLES]
    y = best_v[N_CIRCLES:2*N_CIRCLES]
    r = best_v[2*N_CIRCLES:]
    
    # Strict validity enforcement to satisfy validator tolerances exactly
    for i in range(N_CIRCLES):
        r[i] = min(r[i], x[i], 1.0-x[i], y[i], 1.0-y[i])
        
    for i in range(N_CIRCLES):
        for j in range(i+1, N_CIRCLES):
            d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
            if d < r[i] + r[j]:
                shrink = (r[i] + r[j] - d) / 2.0 + 1e-7
                r[i] = max(0.0, r[i] - shrink)
                r[j] = max(0.0, r[j] - shrink)
                
    centers = np.column_stack([x, y])
    return centers, r, float(np.sum(r))
