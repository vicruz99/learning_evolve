# sol_000023 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a097d99c) state=bc8e03ec sum of radii=0.761775 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(p, n):
    """
    Objective function to maximize radius r while penalizing overlaps and boundary violations.
    p: concatenated array of centers [x0, y0, x1, y1, ...] and radius r
    """
    c = p[:2*n].reshape(n, 2)
    r = p[2*n]
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)  # Ignore self-distances
    
    # Overlap penalty: max(0, 2r - dist)^2
    overlaps = np.maximum(0, 2*r - dists)
    overlap_pen = np.sum(overlaps**2)
    
    # Boundary penalty: max(0, violation)^2
    bound_pen = np.sum(np.maximum(0, r - c[:, 0])**2)
    bound_pen += np.sum(np.maximum(0, r - c[:, 1])**2)
    bound_pen += np.sum(np.maximum(0, c[:, 0] - (1 - r))**2)
    bound_pen += np.sum(np.maximum(0, c[:, 1] - (1 - r))**2)
    
    # Heavy penalty weight ensures constraints are prioritized
    return -r + 5000.0 * (overlap_pen + bound_pen)

def run_packing():
    n = 26
    
    # 1. Initial Hexagonal Grid Placement
    centers = np.zeros((n, 2))
    rows = 5
    y_vals = np.linspace(0.15, 0.85, rows)
    idx = 0
    # Staggered row lengths summing to 26
    row_lengths = [6, 5, 6, 5, 4]
    
    for r_idx in range(rows):
        num = row_lengths[r_idx]
        # Stagger even rows slightly for hexagonal packing
        offset = (r_idx % 2) * 0.08
        x_vals = np.linspace(0.05, 0.95, num) + offset
        for x in x_vals:
            if idx < n:
                centers[idx] = [np.clip(x, 0, 1), y_vals[r_idx]]
                idx += 1
                
    # 2. Setup Optimization
    r_init = 0.09
    x0 = np.concatenate([centers.flatten(), [r_init]])
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)]
    
    # 3. Run L-BFGS-B Optimizer
    res = minimize(
        compute_objective, 
        x0, 
        args=(n,),
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-10}
    )
    
    c_opt = res.x[:2*n].reshape(n, 2)
    r_opt = res.x[2*n]
    
    # 4. Compute tightest valid radius to guarantee strict constraint satisfaction
    dists = np.sqrt(np.sum((c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    min_x = np.min(c_opt[:, 0])
    min_y = np.min(c_opt[:, 1])
    max_x = np.max(c_opt[:, 0])
    max_y = np.max(c_opt[:, 1])
    
    # Radius is limited by nearest neighbor and nearest boundary
    r_feas = min(min_pair_dist / 2.0, min_x, min_y, 1 - max_x, 1 - max_y)
    r_final = max(0.0, r_feas)
    
    radii = np.full(n, r_final)
    sum_radii = n * r_final
    
    return c_opt, radii, sum_radii
