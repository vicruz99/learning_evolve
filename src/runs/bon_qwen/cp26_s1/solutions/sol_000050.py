# sol_000050 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b794a107) state=91f432d7 sum of radii=2.602626 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(vars, n):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(vars[2::3])

def _cons_bound(vars, n):
    """Boundary constraints: circles inside [0,1]x[0,1]"""
    c = vars[:3*n].reshape((n, 3))
    return np.concatenate([c[:,0]-c[:,2], 1-c[:,0]-c[:,2], 
                           c[:,1]-c[:,2], 1-c[:,1]-c[:,2]])

def _cons_overlap(vars, n):
    """Non-overlap constraints: distance between centers >= sum of radii"""
    c = vars[:3*n].reshape((n, 3))
    diff = c[:, np.newaxis, :2] - c[np.newaxis, :, :2]
    dist2 = np.sum(diff**2, axis=2)
    r_sum = c[:, 2, np.newaxis] + c[np.newaxis, :, 2]
    vals = dist2 - r_sum**2
    return vals[np.triu_indices(n, k=1)]

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Try multiple restarts to avoid local minima
    for seed in range(3):
        np.random.seed(seed)
        
        # Initialize in a feasible grid pattern
        grid_x = np.linspace(0.1, 0.9, 6)
        grid_y = np.linspace(0.1, 0.9, 5)
        pts = []
        for y in grid_y:
            for x in grid_x:
                pts.append([x, y])
        
        # Perturb slightly based on seed
        perturbation = np.random.uniform(-0.02, 0.02, size=(n, 2))
        centers_init = np.array(pts[:n]) + perturbation
        centers_init = np.clip(centers_init, 0.05, 0.95)
        radii_init = np.full(n, 0.015)
        
        x0 = np.concatenate([centers_init.ravel(), radii_init])
        
        constraints = [
            {'type': 'ineq', 'fun': _cons_bound, 'args': (n,)},
            {'type': 'ineq', 'fun': _cons_overlap, 'args': (n,)}
        ]
        
        bounds = [(0.0, 1.0) if k % 3 != 2 else (0.0, 0.5) for k in range(3 * n)]
        
        res = minimize(_objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, args=(n,),
                       options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
        
        if -res.fun > best_sum:
            c = res.x[:3*n].reshape((n, 3))[:, :2]
            r = res.x[:3*n].reshape((n, 3))[:, 2]
            # Ensure radii are non-negative
            r = np.maximum(r, 0.0)
            best_sum = np.sum(r)
            best_centers = c
            best_radii = r
            
    return best_centers, best_radii, float(best_sum)
