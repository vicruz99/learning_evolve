# sol_000037 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8e46300b) state=4524ec9a sum of radii=2.573852 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
# Precompute indices for the upper triangle to avoid diagonal (i=i) and lower triangle
_ROW_IDX, _COL_IDX = np.triu_indices(N_CIRCLES, k=1)

def _constraints(v):
    """
    Computes all inequality constraints for the packing problem.
    Constraints must be >= 0 for a valid configuration.
    v: flattened array [x0, y0, r0, x1, y1, r1, ...]
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    # Boundary constraints: 
    # x - r >= 0  => x >= r
    # 1 - x - r >= 0 => x + r <= 1
    # Same for y, and r >= 0
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r,
        r
    ])
    
    # Pairwise non-overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
    X = x[:, np.newaxis]
    Y = y[:, np.newaxis]
    R = r[:, np.newaxis]
    
    dist_sq = (X - X.T)**2 + (Y - Y.T)**2
    r_sum_sq = (R + R.T)**2
    
    # Extract only i < j pairs
    c_pair = dist_sq[_ROW_IDX, _COL_IDX] - r_sum_sq[_ROW_IDX, _COL_IDX]
    
    return np.concatenate([c_bound, c_pair])

def run_packing():
    N = N_CIRCLES
    
    # 1. Initialize with a hexagonal grid for high initial density
    centers = []
    y_step = np.sqrt(3) / 2.0
    count = 0
    for row in range(7):
        for col in range(7):
            if count >= N:
                break
            cx = col * 1.0 + (0.5 if row % 2 == 1 else 0.0)
            cy = row * y_step
            centers.append([cx, cy])
            count += 1
        if count >= N:
            break
            
    centers = np.array(centers[:N])
    # Scale to fit comfortably inside [0,1]^2 with margin
    min_c = centers.min(axis=0)
    max_c = centers.max(axis=0)
    centers = 0.15 + (centers - min_c) / (max_c - min_c) * 0.7
    
    r_init = 0.05 * np.ones(N)
    
    # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
    v0 = np.zeros(3 * N)
    v0[0::3] = centers[:, 0]
    v0[1::3] = centers[:, 1]
    v0[2::3] = r_init
    
    # Bounds: coordinates in [0,1], radii in [0,1]
    bounds = [(0.0, 1.0)] * (3 * N)
    
    def objective(v):
        return -np.sum(v[2::3])
        
    cons = {'type': 'ineq', 'fun': _constraints}
    
    # 2. Optimize using SLSQP
    res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12})
    
    v_opt = res.x
    centers_opt = np.column_stack((v_opt[0::3], v_opt[1::3]))
    radii_opt = v_opt[2::3]
    
    return centers_opt, radii_opt, float(np.sum(radii_opt))
