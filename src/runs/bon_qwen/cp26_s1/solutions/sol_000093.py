# sol_000093 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9fb5006a) state=eff63488 sum of radii=2.458005 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # 1. Initialization: Structured grid with perturbation
    x_grid = np.linspace(0.15, 0.85, 5)
    y_grid = np.linspace(0.15, 0.85, 6)
    centers = []
    for y in y_grid:
        for x in x_grid:
            centers.append([x, y])
            if len(centers) >= n:
                break
        if len(centers) >= n:
            break
    centers = np.array(centers[:n])
    
    # Break symmetry to help optimization escape local minima
    np.random.seed(42)
    centers += np.random.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    r0 = 0.08
    x0 = np.concatenate([centers.flatten(), [r0]])
    
    # Precompute indices for pairwise distance constraints (upper triangle)
    tri_idx = np.triu_indices(n, k=1)
    
    # 2. Optimization Functions
    def objective(vars_):
        # We want to maximize r, so we minimize -r
        return -vars_[-1]
        
    def constr_boundary(vars_):
        r = vars_[-1]
        c = vars_[:-1].reshape(-1, 2)
        # x >= r, x <= 1-r, y >= r, y <= 1-r
        return np.concatenate([
            c[:, 0] - r,
            1.0 - r - c[:, 0],
            c[:, 1] - r,
            1.0 - r - c[:, 1]
        ])
        
    def constr_overlap(vars_):
        r = vars_[-1]
        c = vars_[:-1].reshape(-1, 2)
        # Vectorized pairwise squared distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        # Constraint: dist_sq >= 4*r^2  =>  dist_sq - 4*r^2 >= 0
        return dist_sq[tri_idx] - 4.0 * r**2
        
    cons = [
        {'type': 'ineq', 'fun': constr_boundary},
        {'type': 'ineq', 'fun': constr_overlap}
    ]
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 1.0)]
    
    # 3. Run Optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                   constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
    
    final_r = res.x[-1]
    final_centers = res.x[:-1].reshape(-1, 2)
    radii = np.full(n, final_r)
    
    return final_centers, radii, np.sum(radii)
