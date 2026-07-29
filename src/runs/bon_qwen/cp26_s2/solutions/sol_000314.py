# sol_000314 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1cbfbe8a) state=0c4bf4ee sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # 1. Initialize with a hexagonal lattice pattern
    # Pattern: 6, 5, 6, 5, 4 circles per row
    rows = [6, 5, 6, 5, 4]
    centers = []
    y = 0.08
    dx = 0.16
    dy = dx * np.sqrt(3) / 2
    shift_flag = False
    
    for count in rows:
        x = 0.08
        if shift_flag:
            x += dx / 2
        for _ in range(count):
            centers.append([x, y])
            x += dx
        y += dy
        shift_flag = not shift_flag
        
    centers = np.array(centers)
    radii = np.full(n, 0.08)
    
    # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.concatenate([centers.ravel(), radii])
    
    # Bounds for variables
    lb = [0.0] * (2 * n) + [0.0] * n
    ub = [1.0] * (2 * n) + [0.5] * n
    bounds = list(zip(lb, ub))
    
    # Precompute indices for pairwise constraints
    row_idx, col_idx = np.triu_indices(n, k=1)
    
    def obj(vars):
        # Maximize sum of radii => Minimize negative sum
        return -np.sum(vars[2*n:])
        
    def con_dist(vars):
        # Vectorized distance constraints: dist^2 >= (r_i + r_j)^2
        X = vars[:2*n].reshape((n, 2))
        R = vars[2*n:]
        diff = X[row_idx] - X[col_idx]
        dist_sq = np.sum(diff**2, axis=1)
        r_sum = R[row_idx] + R[col_idx]
        return dist_sq - r_sum**2
        
    def con_boundary(vars):
        # Vectorized boundary constraints
        X = vars[:2*n].reshape((n, 2))
        R = vars[2*n:]
        res = np.empty(4 * n)
        res[0::4] = X[:, 0] - R
        res[1::4] = 1.0 - X[:, 0] - R
        res[2::4] = X[:, 1] - R
        res[3::4] = 1.0 - X[:, 1] - R
        return res
        
    constraints = [
        {'type': 'ineq', 'fun': con_dist},
        {'type': 'ineq', 'fun': con_boundary}
    ]
    
    # 2. Run Optimization
    res = minimize(
        obj, x0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
    )
    
    # 3. Extract and validate results
    final_vars = res.x
    centers = final_vars[:2*n].reshape((n, 2))
    radii = final_vars[2*n:]
    
    # Ensure strict validity within numerical tolerance
    radii = np.maximum(radii, 0.0)
    centers = np.clip(centers, 0.0, 1.0)
    
    return centers, radii, np.sum(radii)
