# sol_000091 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f395aea4) state=74fad748 sum of radii=2.564312 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(v):
    """Objective function: minimize negative sum of radii"""
    return -np.sum(v[2::3])

def _constraints(v):
    """Inequality constraints: non-overlap of circles"""
    N = 26
    r = v[2::3]
    u = v[0::3]
    vc = v[1::3]
    
    # Transform normalized coordinates to actual coordinates
    # x = r + u * (1 - 2r) ensures r <= x <= 1-r when u in [0,1]
    x = r + u * (1.0 - 2.0 * r)
    y = r + vc * (1.0 - 2.0 * r)
    
    # Compute pairwise squared distances and minimum allowed squared distances
    X = x[:, None] - x[None, :]
    Y = y[:, None] - y[None, :]
    R = r[:, None] + r[None, :]
    
    # Only upper triangle is needed for pairwise constraints
    idx = np.triu_indices(N, k=1)
    return (X[idx]**2 + Y[idx]**2) - R[idx]**2

def run_packing():
    N = 26
    best_v = None
    best_sum = -1.0
    
    # Try multiple initializations to escape local optima
    for seed in range(40):
        np.random.seed(seed)
        
        v_init = np.zeros(3 * N)
        
        # Alternate between random and grid-perturbed initializations
        if seed < 20:
            v_init[0::3] = np.random.rand(N)
            v_init[1::3] = np.random.rand(N)
            v_init[2::3] = 0.035
        else:
            # Grid-based initialization (5x5)
            v_init[2::3] = 0.04
            k = 0
            for i in range(5):
                for j in range(5):
                    if k < N:
                        xg = 0.1 + i * 0.2
                        yg = 0.1 + j * 0.2
                        r_val = v_init[2 + 3 * k]
                        v_init[3 * k] = (xg - r_val) / (1.0 - 2.0 * r_val)
                        v_init[3 * k + 1] = (yg - r_val) / (1.0 - 2.0 * r_val)
                        k += 1
            if k < N:
                v_init[3 * k] = 0.5
                v_init[3 * k + 1] = 0.5
                
        # Add small perturbation
        v_init += np.random.randn(3 * N) * 0.02
        
        # Bounds: u, v in [0, 1], r in [small, 0.45]
        bounds = [(0.0, 1.0)] * (2 * N) + [(1e-5, 0.45)] * N
        
        res = minimize(_objective, v_init, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': _constraints},
                       options={'maxiter': 2000, 'ftol': 1e-11})
                       
        if res.success:
            r_cand = res.x[2::3]
            s = np.sum(r_cand)
            if s > best_sum:
                best_sum = s
                best_v = res.x
                
    if best_v is not None:
        r = best_v[2::3].copy()
        u = best_v[0::3]
        vc = best_v[1::3]
        x = r + u * (1.0 - 2.0 * r)
        y = r + vc * (1.0 - 2.0 * r)
        
        # Apply a tiny safety margin to guarantee strict validity
        r *= 0.99995
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
        centers = np.column_stack((x, y))
        return centers, r, np.sum(r)
        
    # Fallback to a safe grid packing
    centers = np.zeros((N, 2))
    r = np.full(N, 0.09)
    k = 0
    for i in range(5):
        for j in range(5):
            if k < N:
                centers[k] = [0.1 + i * 0.2, 0.1 + j * 0.2]
                k += 1
    if k < N:
        centers[k] = [0.5, 0.5]
        r[k] = 0.01
    return centers, r, np.sum(r)
