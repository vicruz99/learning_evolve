# sol_000044 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 5778b268) state=526ed5c8 sum of radii=2.314777 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_initial_centers(n, seed=42):
    """Generates a hexagonal lattice initialization for n circles."""
    np.random.seed(seed)
    centers = []
    r_init = 0.09
    dy = r_init * np.sqrt(3)
    y = r_init
    row_idx = 0
    while y + r_init <= 1.0:
        shift = r_init if row_idx % 2 == 1 else 0.0
        x = r_init + shift
        while x + r_init <= 1.0 and len(centers) < n:
            centers.append([x, y])
            x += 2 * r_init
        y += dy
        row_idx += 1
    # Fallback filler if lattice doesn't yield enough points
    while len(centers) < n:
        centers.append([np.random.rand(), np.random.rand()])
    return np.array(centers[:n])

def compute_penalty_obj(vars, n, mu):
    """Computes objective (negative sum of radii) plus penalty for constraint violations."""
    xs = vars[0::3]
    ys = vars[1::3]
    rs = vars[2::3]
    
    # Objective: maximize sum of radii => minimize negative sum
    obj = -np.sum(rs)
    
    # Penalty accumulator
    p = 0.0
    
    # Boundary violations: circle must be inside [0,1]x[0,1]
    p += np.sum(np.maximum(0.0, rs - xs)**2)
    p += np.sum(np.maximum(0.0, rs - (1.0 - xs))**2)
    p += np.sum(np.maximum(0.0, rs - ys)**2)
    p += np.sum(np.maximum(0.0, rs - (1.0 - ys))**2)
    
    # Overlap violations: dist(i, j) >= r_i + r_j
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = rs[:, None] + rs[None, :]
    viol = np.maximum(0.0, r_sum - dist)
    
    # Only consider upper triangle to avoid double counting
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    p += np.sum(viol[mask]**2)
    
    return obj + mu * p

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses a penalty-based L-BFGS-B optimizer with annealing schedule.
    """
    n = 26
    best_vars = None
    best_score = np.inf
    
    # Try multiple initializations to escape local minima
    for seed in range(5):
        centers_init = get_initial_centers(n, seed=seed + 100)
        # Add small perturbation to break symmetry and explore nearby basins
        centers_init += np.random.normal(0, 0.005, centers_init.shape)
        centers_init = np.clip(centers_init, 0.05, 0.95)
        
        # Initialize variables: [x0, y0, r0, x1, y1, r1, ...]
        vars_init = np.zeros(3 * n)
        vars_init[0::3] = centers_init[:, 0]
        vars_init[1::3] = centers_init[:, 1]
        vars_init[2::3] = 0.08  # Start with a feasible radius
        
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        
        current_vars = vars_init.copy()
        
        # Annealing schedule for penalty weight mu
        for mu in [200, 1000, 5000]:
            res = minimize(
                compute_penalty_obj,
                current_vars,
                args=(n, mu),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-9}
            )
            current_vars = res.x
            
        final_score = compute_penalty_obj(current_vars, n, 5000)
        if final_score < best_score:
            best_score = final_score
            best_vars = current_vars.copy()
            
    # Extract optimal variables
    xs = best_vars[0::3]
    ys = best_vars[1::3]
    rs = best_vars[2::3]
    
    centers = np.column_stack((xs, ys))
    radii = rs
    
    # Post-processing: Calculate maximum valid scaling factor to ensure strict feasibility
    min_viol = 1.0
    if np.any(radii > 1e-9):
        # Boundary constraints
        min_viol = min(min_viol, np.min(xs / radii), np.min((1.0 - xs) / radii), 
                       np.min(ys / radii), np.min((1.0 - ys) / radii))
        
        # Overlap constraints
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        r_sum = radii[:, None] + radii[None, :]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        ratios = dist[mask] / r_sum[mask]
        if len(ratios) > 0:
            min_viol = min(min_viol, np.min(ratios))
            
    # Apply safety margin to satisfy validator's 1e-12 tolerance
    scale = min(1.0, min_viol) * 0.9999
    radii *= scale
    
    return centers, radii, float(np.sum(radii))
