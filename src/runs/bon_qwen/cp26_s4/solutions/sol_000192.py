# sol_000192 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bafdbd7e) state=e49fd6f6 sum of radii=2.511709 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars):
    """Objective: minimize negative sum of radii"""
    n = len(vars) // 3
    radii = vars[2*n:]
    return -np.sum(radii)

def compute_constraints(vars):
    """Compute all inequality constraints: overlaps >= 0, boundaries >= 0"""
    n = len(vars) // 3
    centers = vars[:2*n].reshape((n, 2))
    radii = vars[2*n:]
    
    cx = centers[:, 0]
    cy = centers[:, 1]
    
    # Pairwise squared distances
    diff_x = cx[:, None] - cx[None, :]
    diff_y = cy[:, None] - cy[None, :]
    dist_sq = diff_x**2 + diff_y**2
    
    # Squared sums of radii
    r_sum = radii[:, None] + radii[None, :]
    r_sum_sq = r_sum**2
    
    # Overlap constraints: dist^2 - (r1+r2)^2 >= 0
    tri_idx = np.triu_indices(n, k=1)
    overlaps = dist_sq[tri_idx] - r_sum_sq[tri_idx]
    
    # Boundary constraints: x-r >=0, 1-x-r >=0, y-r >=0, 1-y-r >=0
    bounds = np.concatenate([
        cx - radii,
        1.0 - cx - radii,
        cy - radii,
        1.0 - cy - radii
    ])
    
    return np.concatenate([overlaps, bounds])

def run_packing():
    n = 26
    best_sum_r = -1.0
    best_centers = None
    best_radii = None
    
    # Variable bounds: x, y in [0, 1], r in [1e-5, 0.5]
    bounds = []
    for i in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (1e-5, 0.5)])
        
    cons = ({'type': 'ineq', 'fun': compute_constraints})
    
    # Run multiple optimizations from different starting points
    for seed in range(15):
        np.random.seed(seed)
        
        # Hexagonal-ish grid initialization
        base = np.linspace(0.12, 0.88, 6)
        grid_x, grid_y = np.meshgrid(base, base)
        init_pos = np.vstack([grid_x.ravel(), grid_y.ravel()]).T[:n]
        
        # Add noise to break symmetry and encourage diverse packings
        noise = np.random.uniform(-0.02, 0.02, init_pos.shape)
        init_pos = np.clip(init_pos + noise, 0.05, 0.95)
        
        # Start with small feasible radii to ensure initial constraint satisfaction
        init_radii = np.full(n, 0.02)
        x0 = np.hstack([init_pos.ravel(), init_radii])
        
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds, 
                          constraints=cons, options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False})
            
            if res.success:
                cur_sum = -res.fun
                if cur_sum > best_sum_r:
                    best_sum_r = cur_sum
                    best_centers = res.x[:2*n].reshape((n, 2))
                    best_radii = res.x[2*n:]
        except Exception:
            continue
            
    # Fallback to ensure valid return in case of unexpected failures
    if best_centers is None:
        best_centers = np.random.uniform(0.15, 0.85, (n, 2))
        best_radii = np.full(n, 0.015)
        best_sum_r = np.sum(best_radii)
        
    return best_centers, best_radii, best_sum_r
