# sol_000070 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3353d097) state=bd1a778f sum of radii=2.480249 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def get_boundary_constraints(centers, r):
    """Computes boundary constraint values: x-r, y-r, 1-x-r, 1-y-r >= 0"""
    n = centers.shape[0]
    con = np.empty(4 * n)
    con[:n] = centers[:, 0] - r
    con[n:2*n] = centers[:, 1] - r
    con[2*n:3*n] = 1.0 - centers[:, 0] - r
    con[3*n:] = 1.0 - centers[:, 1] - r
    return con

def get_overlap_constraints(centers, r):
    """Computes overlap constraint values: dist_ij - 2r >= 0"""
    return pdist(centers) - 2.0 * r

def objective_function(x):
    """Maximize radius (minimize negative radius)"""
    return -x[-1]

def constraint_boundary(x):
    r = x[-1]
    centers = x[:-1].reshape(-1, 2)
    return get_boundary_constraints(centers, r)

def constraint_overlap(x):
    r = x[-1]
    centers = x[:-1].reshape(-1, 2)
    return get_overlap_constraints(centers, r)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Generate initial hexagonal lattice configuration
    r_init = 0.101
    centers_init = []
    cols, rows = 5, 6
    
    for r_idx in range(rows):
        for c_idx in range(cols):
            if len(centers_init) >= n:
                break
            x = c_idx * 2 * r_init + (r_idx % 2) * r_init
            y = r_idx * np.sqrt(3) * r_init
            centers_init.append([x, y])
        if len(centers_init) >= n:
            break
            
    centers_init = np.array(centers_init[:n])
    
    # Normalize to center within the unit square
    min_c = centers_init.min(axis=0)
    max_c = centers_init.max(axis=0)
    span = np.max(max_c - min_c)
    scale = 0.85 / span
    centers_init = (centers_init - min_c) * scale + (1.0 - 0.85) / 2.0
    
    # 2. Setup optimization variables and bounds
    x0 = np.concatenate([centers_init.flatten(), [r_init]])
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.2)]
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # 3. Run optimizer
    res = minimize(
        objective_function, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
    )
    
    final_centers = res.x[:-1].reshape(-1, 2)
    final_r = res.x[-1]
    
    # 4. Post-processing to ensure strict feasibility
    # Check and enforce boundary constraints
    for i in range(n):
        x, y = final_centers[i]
        final_r = min(final_r, x, y, 1.0 - x, 1.0 - y)
        
    # Check and enforce non-overlap constraints
    dists = pdist(final_centers)
    min_dist = np.min(dists)
    if min_dist < 2.0 * final_r - 1e-10:
        final_r = (min_dist * 0.9999) / 2.0
        
    radii = np.full(n, final_r)
    total_sum = np.sum(radii)
    
    return final_centers, radii, total_sum
