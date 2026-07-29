# sol_000082 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000001 (state 1501c8b5) state=4263120b sum of radii=0.359026 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(v):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[2*N_CIRCLES:3*N_CIRCLES])

def constraint_func(v):
    """
    Computes all inequality constraints:
    1. Boundary: center +/- radius within [0, 1]
    2. Pairwise: squared distance >= squared sum of radii
    Returns a flat array where each element >= 0 indicates a satisfied constraint.
    """
    n = N_CIRCLES
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:3*n]
    
    cons = []
    # Boundary constraints
    cons.append(x - r)
    cons.append(1.0 - x - r)
    cons.append(y - r)
    cons.append(1.0 - y - r)
    
    # Pairwise non-overlap constraints using squared distances to avoid sqrt singularities
    X = x[:, np.newaxis] - x[np.newaxis, :]
    Y = y[:, np.newaxis] - y[np.newaxis, :]
    R = r[:, np.newaxis] + r[np.newaxis, :]
    dist_sq = X**2 + Y**2
    r_sum_sq = R**2
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    cons.append((dist_sq - r_sum_sq)[mask])
    
    return np.concatenate(cons)

def generate_hex_init(n, r_start, seed):
    """Generates a hexagonal lattice initialization with controlled perturbation."""
    np.random.seed(seed)
    centers = []
    y = r_start
    row = 0
    while len(centers) < n:
        x = r_start if row % 2 == 0 else 2 * r_start
        while x <= 1 - r_start and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_start
        y += r_start * np.sqrt(3)
        row += 1
    centers = np.array(centers[:n])
    centers += np.random.uniform(-0.005, 0.005, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    return centers

def run_packing():
    n = N_CIRCLES
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = 0.0
    best_v = None
    
    # Multiple restarts from hexagonal lattice configurations
    for seed in range(20):
        init_centers = generate_hex_init(n, r_start=0.09, seed=seed)
        # Start with a feasible radius smaller than lattice spacing to guarantee initial validity
        v0 = np.concatenate([init_centers[:, 0], init_centers[:, 1], np.full(n, 0.07)])
        
        try:
            res = minimize(objective_func, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        # Fallback initialization if optimization fails completely
        best_v = np.concatenate([np.random.rand(2*n)*0.8+0.1, np.full(n, 0.05)])
        
    # Refinement phase: perturb best solution slightly and re-optimize to escape shallow local minima
    for _ in range(8):
        v0_ref = best_v + np.random.uniform(-0.0005, 0.0005, best_v.shape)
        v0_ref[:n] = np.clip(v0_ref[:n], 0.01, 0.99)
        v0_ref[n:2*n] = np.clip(v0_ref[n:2*n], 0.01, 0.99)
        v0_ref[2*n:] = np.clip(v0_ref[2*n:], 0.001, 0.4)
        
        try:
            res = minimize(objective_func, v0_ref, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_v = res.x.copy()
        except Exception:
            pass
            
    centers = best_v[:2*n].reshape(n, 2)
    radii = best_v[2*n:]
    
    # Strict validity enforcement to guarantee constraints pass validation tolerance
    for _ in range(200):
        changed = False
        for i in range(n):
            max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
            for j in range(n):
                if i == j: continue
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                allowed = d - radii[j]
                if allowed < max_r:
                    max_r = allowed
            if max_r < radii[i] - 1e-10:
                radii[i] = max(0.0, max_r)
                changed = True
        if not changed:
            break
            
    # Final safety shrink to guarantee 1e-12 tolerance margin
    radii *= 0.9999999998
    
    return centers, radii, np.sum(radii)
