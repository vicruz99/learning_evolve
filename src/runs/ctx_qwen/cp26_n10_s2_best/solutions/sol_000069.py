# sol_000069 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000026 (state f081a56f) state=57a9ba5b sum of radii=1.057315 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_fun(v, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraints_fun(v, n, pair_i, pair_j):
    """
    Vectorized constraint function.
    Returns an array where all elements must be >= 0.
    """
    cx = v[:n]
    cy = v[n:2*n]
    cr = v[2*n:]
    
    c = []
    # Boundary constraints
    c.append(cx - cr)           # x >= r
    c.append(1 - cx - cr)       # x + r <= 1
    c.append(cy - cr)           # y >= r
    c.append(1 - cy - cr)       # y + r <= 1
    
    # Pairwise non-overlap constraints
    xi = cx[pair_i]
    xj = cx[pair_j]
    yi = cy[pair_i]
    yj = cy[pair_j]
    ri = cr[pair_i]
    rj = cr[pair_j]
    
    dx = xi - xj
    dy = yi - yj
    dist = np.sqrt(dx*dx + dy*dy)
    c.append(dist - (ri + rj))  # dist >= r_i + r_j
    
    return np.concatenate(c)

def get_init_config(n, method='hex', seed=None):
    """Generates initial center positions and safe radii."""
    if seed is not None:
        np.random.seed(seed)
        
    centers = np.zeros((n, 2))
    
    if method == 'hex':
        pts = []
        r_est = 0.09
        y = r_est
        row = 0
        while len(pts) < n + 10:
            x = r_est + (row % 2) * r_est
            while x < 1 - r_est:
                pts.append([x, y])
                x += 2 * r_est
            y += np.sqrt(3) * r_est
            row += 1
        centers = np.array(pts[:n])
        
    elif method == 'grid':
        pts = []
        for i in range(5):
            for j in range(6):
                if len(pts) < n:
                    pts.append([0.05 + j*0.17, 0.05 + i*0.19])
        centers = np.array(pts)
        
    elif method == 'random':
        centers = np.random.uniform(0.05, 0.95, (n, 2))
        
    # Add controlled jitter to break symmetry
    centers += np.random.uniform(-0.005, 0.005, size=centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    # Compute dynamically safe initial radii based on point density
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    min_d = np.min(dists)
    r_init = np.full(n, min_d * 0.45)
    
    return centers, r_init

def run_packing():
    n = 26
    pair_i, pair_j = np.triu_indices(n, k=1)
    
    # Variable bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    cons_dict = {
        'type': 'ineq', 
        'fun': constraints_fun, 
        'args': (n, pair_i, pair_j)
    }
    
    best_sum = -1.0
    best_sol = None
    
    # Multiple restart strategies to escape local minima
    strategies = [
        ('hex', 0), ('hex', 1), ('hex', 2),
        ('grid', 0), ('grid', 1),
        ('random', 0), ('random', 1), ('random', 2), ('random', 3)
    ]
    
    for method, seed in strategies:
        centers0, r0 = get_init_config(n, method, seed)
        v0 = np.concatenate([centers0.ravel(), r0])
        
        try:
            res = minimize(
                objective_fun, v0, args=(n,), 
                method='SLSQP', bounds=bounds, 
                constraints=cons_dict, 
                options={'maxiter': 10000, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.success:
                cur_sum = -res.fun
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_sol = res.x.copy()
        except Exception:
            pass
            
    # Fallback if all optimizations fail (unlikely)
    if best_sol is None:
        centers0, r0 = get_init_config(n, 'hex', 0)
        best_sol = np.concatenate([centers0.ravel(), r0])
        best_sum = np.sum(r0)
        
    centers = best_sol[:2*n].reshape(n, 2)
    radii = best_sol[2*n:].copy()
    
    # Strict post-processing to guarantee validator tolerance is met
    for _ in range(30):
        changed = False
        for i in range(n):
            # Enforce boundary constraints
            r_lim = min(centers[i,0], 1-centers[i,0], centers[i,1], 1-centers[i,1])
            if radii[i] > r_lim:
                radii[i] = r_lim
                changed = True
                
            # Enforce non-overlap constraints
            for j in range(n):
                if i == j: continue
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                r_req = d - radii[j]
                if radii[i] > r_req + 1e-9:
                    radii[i] = max(0.0, r_req - 1e-9)
                    changed = True
        if not changed:
            break
            
    # Final safety margin
    radii *= 0.9999999
    centers = np.clip(centers, 1e-9, 1.0 - 1e-9)
    
    return centers, radii, float(np.sum(radii))
