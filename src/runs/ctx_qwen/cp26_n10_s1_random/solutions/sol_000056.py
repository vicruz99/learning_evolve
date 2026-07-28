# sol_000056 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state 6f2d6856) state=ac9b2da0 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def obj(vars):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def cons(vars):
    """
    Constraints: 
    1. Boundary constraints (4*N)
    2. Non-overlap constraints (N*(N-1)/2)
    Returns array of values that must be >= 0.
    """
    n = N
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    b_cons = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap constraints: dist_sq >= (r_i + r_j)^2
    # Vectorized computation for efficiency
    cx = x[:, np.newaxis] - x[np.newaxis, :]
    cy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = cx**2 + cy**2
    np.fill_diagonal(dist_sq, 1e6)  # Ignore diagonal
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    triu_idx = np.triu_indices(n, k=1)
    p_cons = dist_sq[triu_idx] - r_sum_sq[triu_idx]
    
    return np.concatenate([b_cons, p_cons])

def run_packing():
    n = N
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    constraint_dict = {'type': 'ineq', 'fun': cons}
    
    best_sum = -np.inf
    best_vars = None
    rng = np.random.default_rng(42)
    
    configs = []
    
    # 1. Hexagonal lattice initialization
    r_init = 0.085
    pts = []
    row = 0
    y = r_init
    dy = np.sqrt(3) * r_init
    while y + r_init <= 1.0:
        shift = 0.0 if row % 2 == 0 else r_init
        x = r_init + shift
        while x + r_init <= 1.0:
            pts.append([x, y])
            x += 2 * r_init
        y += dy
        row += 1
        if len(pts) >= n:
            break
            
    pts = np.array(pts[:n])
    while len(pts) < n:
        pts = np.vstack([pts, [[0.5, 0.5]]])
        
    x0_hex = np.zeros(3 * n)
    x0_hex[0::3] = pts[:, 0]
    x0_hex[1::3] = pts[:, 1]
    x0_hex[2::3] = r_init
    configs.append(x0_hex)
    
    # 2. Perturbed hexagonal lattice
    configs.append(x0_hex + rng.uniform(-0.02, 0.02, 3 * n))
    
    # 3. Random initializations to escape local minima
    for _ in range(4):
        x0 = np.zeros(3 * n)
        x0[0::3] = rng.uniform(0.15, 0.85, n)
        x0[1::3] = rng.uniform(0.15, 0.85, n)
        x0[2::3] = 0.07
        configs.append(x0)
        
    # Run optimization from each configuration
    for x0 in configs:
        try:
            res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraint_dict,
                           options={'maxiter': 1500, 'ftol': 1e-10})
            if np.isfinite(res.fun):
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_vars = res.x
        except Exception:
            continue
            
    if best_vars is None:
        best_vars = configs[0]
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    
    # Post-processing repair to guarantee strict validity
    radii = np.maximum(radii, 0.0)
    
    # Scale down radii if any pairwise overlaps exist due to numerical tolerance
    scale = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            req = radii[i] + radii[j]
            if req > 1e-9 and dist < req:
                scale = min(scale, dist / req)
                
    if scale < 1.0:
        radii *= scale
        
    # Clip centers to ensure they strictly respect boundary constraints
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
    return centers, radii, float(best_sum)
