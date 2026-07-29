# sol_000047 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state ce356e52) state=e8f83ee3 sum of radii=2.608851 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars_vec):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    return -np.sum(vars_vec[2::3])

def constraint_func(vars_vec):
    """
    Constraint function: ensures circles are inside [0,1]^2 and do not overlap.
    Returns array of constraint values >= 0.
    """
    x = vars_vec[0::3]
    y = vars_vec[1::3]
    r = vars_vec[2::3]
    
    # Wall constraints: r <= x <= 1-r, r <= y <= 1-r
    c_wall = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, None] + r[None, :]
    r_sum_sq = r_sum**2
    
    # Only upper triangular part (i < j) is needed
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c_pair = dist_sq[mask] - r_sum_sq[mask]
    
    return np.concatenate([c_wall, c_pair])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = -np.inf
    best_vars = None
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattice initialization (dense packing structure)
    pts_hex = []
    r_est = 0.092
    y_pos = r_est
    row = 0
    while len(pts_hex) < N:
        x_pos = r_est + (row % 2) * r_est
        while x_pos <= 1.0 - r_est and len(pts_hex) < N:
            pts_hex.append([x_pos, y_pos, r_est])
            x_pos += 2.0 * r_est
        y_pos += np.sqrt(3.0) * r_est
        row += 1
    inits.append(np.array(pts_hex).flatten())
    
    # 2. Square grid 5x5 + center
    pts_sq = []
    for i in range(5):
        for j in range(5):
            pts_sq.append([0.1 + i * 0.2, 0.1 + j * 0.2, 0.09])
    pts_sq.append([0.5, 0.5, 0.05])
    inits.append(np.array(pts_sq).flatten())
    
    # 3-15. Randomized initializations with controlled bounds
    for seed in range(13):
        np.random.seed(seed + 2024)
        x_r = np.random.uniform(0.15, 0.85, N)
        y_r = np.random.uniform(0.15, 0.85, N)
        r_r = np.full(N, 0.075)
        inits.append(np.column_stack((x_r, y_r, r_r)).flatten())
        
    # Phase 1: Multi-restart optimization
    for x0 in inits:
        # Add slight perturbation to break exact symmetries
        x0_pert = x0 + np.random.uniform(-0.012, 0.012, len(x0))
        x0_pert = np.clip(x0_pert, 0.01, 0.99)
        
        try:
            res = minimize(objective, x0_pert, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraint_func},
                           options={'maxiter': 5000, 'ftol': 1e-12})
            
            if res.success:
                val = -res.fun
                c_val = constraint_func(res.x)
                # Accept if constraints are satisfied and improves best sum
                if np.min(c_val) >= -1e-7 and val > best_sum:
                    best_sum = val
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local refinement on the best solution found
    if best_vars is not None:
        for _ in range(6):
            # Perturb centers and radii slightly to escape shallow local minima
            x_ref = best_vars + np.random.uniform(-0.004, 0.004, len(best_vars))
            x_ref = np.clip(x_ref, 0.0, 0.99)
            
            try:
                res_ref = minimize(objective, x_ref, method='SLSQP', bounds=bounds,
                                   constraints={'type': 'ineq', 'fun': constraint_func},
                                   options={'maxiter': 4000, 'ftol': 1e-12})
                
                if res_ref.success:
                    val_ref = -res_ref.fun
                    c_val_ref = constraint_func(res_ref.x)
                    if np.min(c_val_ref) >= -1e-7 and val_ref > best_sum:
                        best_sum = val_ref
                        best_vars = res_ref.x.copy()
            except Exception:
                continue
                
    # Extract and return best valid solution
    if best_vars is not None:
        centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
        radii = best_vars[2::3]
        radii = np.maximum(radii, 0.0)
        return centers, radii, float(np.sum(radii))
        
    # Fallback: Valid but suboptimal grid packing
    x_coords = np.linspace(0.1, 0.9, 5)
    y_coords = np.linspace(0.1, 0.9, 5)
    centers = []
    radii = []
    for y in y_coords:
        for x in x_coords:
            centers.append([x, y])
            radii.append(0.1)
    centers.append([0.5, 0.5])
    radii.append(0.01)
    return np.array(centers), np.array(radii), float(np.sum(radii))
