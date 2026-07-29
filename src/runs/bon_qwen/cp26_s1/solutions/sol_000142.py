# sol_000142 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 68244382) state=ece3d997 sum of radii=0.000033 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def loss_function(params, n, lam):
    """
    Objective function with penalty terms.
    params: array of shape (3n) containing [x0, y0, r0, x1, y1, r1, ...]
    n: number of circles
    lam: penalty weight
    """
    xs = params[0::3]
    ys = params[1::3]
    rs = params[2::3]
    
    # Primary objective: maximize sum of radii -> minimize negative sum
    val = -np.sum(rs)
    
    # Boundary penalties: ensure circles are inside [0,1]x[0,1]
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => x + r - 1 <= 0
    # y - r >= 0  => r - y <= 0
    # y + r <= 1  => y + r - 1 <= 0
    # r >= 0      => -r <= 0
    val += lam * np.sum(np.maximum(0.0, rs - xs)**2)
    val += lam * np.sum(np.maximum(0.0, xs + rs - 1.0)**2)
    val += lam * np.sum(np.maximum(0.0, rs - ys)**2)
    val += lam * np.sum(np.maximum(0.0, ys + rs - 1.0)**2)
    val += lam * np.sum(np.maximum(0.0, -rs)**2)
    
    # Overlap penalties: ensure distance >= r_i + r_j
    # dist_ij - r_i - r_j >= 0 => r_i + r_j - dist_ij <= 0
    centers = np.column_stack((xs, ys))
    # Compute pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    rs_sum = rs[:, None] + rs[None, :]
    
    # Violations where sum of radii exceeds distance
    violations = np.maximum(0.0, rs_sum - dists)
    # Sum over all pairs, divide by 2 to avoid double counting
    val += lam * 0.5 * np.sum(violations**2)
    
    return val

def run_packing():
    n = 26
    best_params = None
    best_score = -np.inf
    
    # Box constraints for optimization
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n

    # Try multiple restarts to find a good local optimum
    for seed in range(5):
        np.random.seed(seed)
        
        # Initialize centers randomly in the central region to avoid boundaries initially
        centers = np.random.uniform(0.2, 0.8, (n, 2))
        radii = np.ones(n) * 0.02
        
        params = np.zeros(3 * n)
        params[0::3] = centers[:, 0]
        params[1::3] = centers[:, 1]
        params[2::3] = radii
        
        current_params = params.copy()
        
        # Annealing schedule for penalty weight
        # Start low to allow exploration, increase to enforce constraints
        for lam in [1.0, 10.0, 100.0, 500.0, 2000.0, 10000.0, 50000.0]:
            res = minimize(loss_function, current_params, args=(n, lam), 
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 1500, 'ftol': 1e-12, 'gtol': 1e-8})
            current_params = res.x
            
            # Track the best valid configuration found so far
            s_r = np.sum(current_params[2::3])
            if s_r > best_score:
                best_score = s_r
                best_params = current_params.copy()

    # Final refinement on the best found configuration
    if best_params is not None:
        current_params = best_params
        # High penalty to ensure strict validity
        for lam in [100000.0, 200000.0]:
            res = minimize(loss_function, current_params, args=(n, lam), 
                           method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 2000, 'ftol': 1e-12})
            current_params = res.x
        
        centers_opt = np.column_stack((current_params[0::3], current_params[1::3]))
        radii_opt = current_params[2::3]
    else:
        # Fallback initialization
        centers_opt = np.random.uniform(0.2, 0.8, (n, 2))
        radii_opt = np.ones(n) * 0.01

    # Post-processing: Strict enforcement of constraints
    # 1. Boundary constraints
    for i in range(n):
        x, y = centers_opt[i]
        r = radii_opt[i]
        # Radius cannot exceed distance to any boundary
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if r > max_r:
            radii_opt[i] = max_r

    # 2. Non-overlap constraints
    # Iteratively resolve any remaining overlaps by shrinking radii
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers_opt[i] - centers_opt[j])
                min_dist = radii_opt[i] + radii_opt[j]
                if d < min_dist - 1e-12:
                    # Scale down radii to just touch
                    scale = d / min_dist
                    radii_opt[i] *= scale
                    radii_opt[j] *= scale
                    changed = True
        if not changed:
            break

    # Ensure no negative radii due to numerical issues
    radii_opt = np.maximum(radii_opt, 0.0)

    sum_radii = np.sum(radii_opt)
    return centers_opt, radii_opt, sum_radii
