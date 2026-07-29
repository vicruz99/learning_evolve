# sol_000038 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 16d9c155) state=d380fe34 sum of radii=2.580617 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
DIM = 2

def objective(v, n):
    """Minimize negative sum of radii => Maximize sum of radii."""
    return -np.sum(v[n*DIM:])

def constraint_fn(v, n):
    """
    Returns array of constraint values >= 0.
    Includes boundary and non-overlap constraints.
    """
    c = v[:n*DIM].reshape(n, DIM)
    r = v[n*DIM:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    b = np.concatenate([c[:, 0] - r, 1 - c[:, 0] - r, c[:, 1] - r, 1 - c[:, 1] - r])
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    o = dists - r[:, np.newaxis] - r[np.newaxis, :]
    
    # Extract upper triangle to avoid duplicate pair constraints
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    o = o[mask]
    
    return np.concatenate([b, o])

def run_packing():
    n = N_CIRCLES
    n_vars = n * (DIM + 1)
    bounds = [(0.0, 1.0)] * (n * DIM) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_fn, 'args': (n,)}
    
    best_val = -np.inf
    best_res = None
    
    # Multiple restarts to avoid local minima
    for seed in range(5):
        rng = np.random.default_rng(seed)
        v0 = np.zeros(n_vars)
        idx = 0
        
        # Structured initial guess: 5x5 grid + 1 center circle
        for i in range(5):
            for j in range(5):
                v0[idx] = 0.1 + j * 0.2
                v0[idx+1] = 0.1 + i * 0.2
                v0[idx+2] = 0.09
                idx += 3
        v0[idx] = 0.5
        v0[idx+1] = 0.5
        v0[idx+2] = 0.05
        idx += 3
        
        # Add random perturbation
        v0 += rng.normal(0, 0.02, n_vars)
        
        # Optimize
        res = minimize(objective, v0, args=(n,), method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 10000, 'ftol': 1e-10, 'disp': False})
                       
        current_val = -res.fun
        if current_val > best_val:
            best_val = current_val
            best_res = res
            
    x_opt = best_res.x[:n*DIM].reshape(n, DIM)
    r_opt = best_res.x[n*DIM:]
    
    # Safety clamping to ensure strict validity
    r_opt = np.maximum(r_opt, 1e-7)
    x_opt = np.clip(x_opt, 1e-7, 1.0 - 1e-7)
    
    return x_opt, r_opt, np.sum(r_opt)
