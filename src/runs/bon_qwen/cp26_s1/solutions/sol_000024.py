# sol_000024 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 27de0ea1) state=cd002c42 sum of radii=2.516856 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

NUM_CIRCLES = 26

def generate_hex_init(n):
    """Generate a hexagonal lattice initialization scaled to the unit square."""
    pts = []
    for i in range(8):
        for j in range(8):
            x = j + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2.0
            pts.append([x, y])
        if len(pts) >= n:
            break
    pts = np.array(pts[:n])
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    range_val = maxs - mins
    if np.any(range_val == 0):
        range_val += 1e-12
    pts = (pts - mins) / range_val * 0.9 + 0.05
    return pts

def generate_random_init(n, rng):
    """Generate a random non-overlapping initialization."""
    pts = []
    for _ in range(n):
        attempts = 0
        while attempts < 1000:
            p = rng.uniform(0.1, 0.9, 2)
            ok = True
            for q in pts:
                if np.hypot(p[0]-q[0], p[1]-q[1]) < 0.12:
                    ok = False
                    break
            if ok:
                pts.append(p)
                break
            attempts += 1
    return np.array(pts)

def compute_constraints(p, n):
    """Compute all boundary and non-overlap constraints as an array >= 0."""
    X, Y, R = p[::3], p[1::3], p[2::3]
    c = []
    # Boundary constraints: center - radius >= 0 and 1 - center - radius >= 0
    c.extend(X - R)
    c.extend(1.0 - X - R)
    c.extend(Y - R)
    c.extend(1.0 - Y - R)
    
    # Pairwise separation constraints: distance - (r_i + r_j) >= 0
    X_diff = X[:, None] - X[None, :]
    Y_diff = Y[:, None] - Y[None, :]
    R_sum = R[:, None] + R[None, :]
    dists = np.sqrt(X_diff**2 + Y_diff**2)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.extend((dists - R_sum)[mask].flatten())
    return np.array(c)

def objective_func(p, n):
    """Negative sum of radii to be minimized."""
    return -np.sum(p[2::3])

def run_packing():
    n = NUM_CIRCLES
    best_sum = -1.0
    best_p = None
    rng = np.random.RandomState(2023)
    
    # Prepare initial configurations
    inits = [generate_hex_init(n)]
    for _ in range(3):
        inits.append(generate_random_init(n, rng))
        
    bounds = [(0.0, 1.0)]*(2*n) + [(1e-6, 0.5)]*n
    cons = {'type': 'ineq', 'fun': lambda p: compute_constraints(p, n)}
    
    # Initial optimization runs
    for init_centers in inits:
        p0 = np.zeros(n*3)
        p0[::3] = init_centers[:, 0]
        p0[1::3] = init_centers[:, 1]
        p0[2::3] = 0.06
        
        res = minimize(objective_func, p0, args=(n,), method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_p = res.x.copy()
            
    # Local perturbation and refinement phase to escape local optima
    if best_p is not None:
        for _ in range(5):
            p_try = best_p + rng.normal(0, 1e-4, size=best_p.shape)
            p_try[::3] = np.clip(p_try[::3], 0.0, 1.0)
            p_try[1::3] = np.clip(p_try[1::3], 0.0, 1.0)
            p_try[2::3] = np.maximum(p_try[2::3], 1e-6)
            
            res = minimize(objective_func, p_try, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-10})
            
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_p = res.x.copy()
                
        # Final high-precision polish
        res = minimize(objective_func, best_p, args=(n,), method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
        best_p = res.x
        best_sum = -res.fun
        
    centers = np.column_stack((best_p[::3], best_p[1::3]))
    radii = best_p[2::3]
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(best_sum)
