# sol_000048 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state ce356e52) state=89d5cf3a sum of radii=2.621447 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_full(vars_vec):
    """Objective: minimize negative sum of radii."""
    n = 26
    return -np.sum(vars_vec[2::3])

def constraints_full(vars_vec):
    """Inequality constraints: boundary containment and non-overlap."""
    n = 26
    x = vars_vec[0::3]
    y = vars_vec[1::3]
    r = vars_vec[2::3]
    
    c = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise non-overlap: dist_sq >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, None] + r[None, :]
    r_sum_sq = r_sum**2
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.append(dist_sq[mask] - r_sum_sq[mask])
    
    return np.concatenate(c)

def get_max_radii_for_centers(centers):
    """Given fixed centers, compute optimal feasible radii and their negative sum."""
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    
    # Distance to boundaries
    dist_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists, axis=1)
    
    # Optimal radius is limited by neighbors and boundaries
    radii = np.minimum(dist_bound, min_dist / 2.0)
    radii = np.maximum(radii, 0.0)
    
    return -np.sum(radii), radii

def center_objective(centers_flat):
    """Objective for Phase 1: minimize negative sum of radii for fixed centers."""
    centers = centers_flat.reshape(26, 2)
    obj_val, _ = get_max_radii_for_centers(centers)
    return obj_val

def run_packing():
    n = 26
    bounds_full = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    best_sum = -np.inf
    best_vars = None
    
    # Generate initial center configurations for Phase 1
    inits_centers = []
    
    # 1. Hexagonal lattice
    pts = []
    r_est = 0.09
    dy = np.sqrt(3) * r_est
    y = r_est
    row = 0
    while len(pts) < n:
        shift = 0.0 if row % 2 == 0 else r_est
        x = r_est + shift
        while x <= 1.0 - r_est and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r_est
        y += dy
        row += 1
    inits_centers.append(np.array(pts[:n]).flatten())
    
    # 2. 5x5 Grid + Center
    grid = np.array([[0.1 + i*0.2, 0.1 + j*0.2] for i in range(5) for j in range(5)])
    grid = np.vstack([grid, [0.5, 0.5]])
    inits_centers.append(grid.flatten())
    
    # 3. Perturbed versions to escape symmetry
    for seed in range(4):
        np.random.seed(seed)
        base = inits_centers[seed % 2]
        pert = base.reshape(n, 2) + np.random.uniform(-0.04, 0.04, (n, 2))
        pert = np.clip(pert, 0.02, 0.98)
        inits_centers.append(pert.flatten())
        
    # Phase 1: Optimize centers only
    best_center_sum = -np.inf
    best_init_full = None
    
    for init_c in inits_centers:
        res = minimize(center_objective, init_c, method='Nelder-Mead', 
                       options={'maxiter': 8000, 'xatol': 1e-7, 'fatol': 1e-7})
        curr_neg_sum, curr_radii = get_max_radii_for_centers(res.x.reshape(n, 2))
        curr_sum = -curr_neg_sum
        
        if curr_sum > best_center_sum:
            best_center_sum = curr_sum
            # Prepare full variables for Phase 2
            full_vars = np.zeros(3 * n)
            full_vars[0::3] = res.x.reshape(n, 2)[:, 0]
            full_vars[1::3] = res.x.reshape(n, 2)[:, 1]
            full_vars[2::3] = curr_radii
            best_init_full = full_vars
            
    # Phase 2: Refine with SLSQP on full variables
    if best_init_full is not None:
        # Shrink radii slightly to ensure strict feasibility for SLSQP start
        best_init_full[2::3] *= 0.95
        best_init_full[0::3] = np.clip(best_init_full[0::3], 0.01, 0.99)
        best_init_full[1::3] = np.clip(best_init_full[1::3], 0.01, 0.99)
        
        try:
            res_full = minimize(objective_full, best_init_full, method='SLSQP', bounds=bounds_full,
                                constraints={'type': 'ineq', 'fun': constraints_full},
                                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            if res_full.success:
                c_val = constraints_full(res_full.x)
                if np.min(c_val) >= -1e-6:
                    best_vars = res_full.x
                    best_sum = -res_full.fun
                else:
                    best_vars = best_init_full
                    best_sum = best_center_sum
            else:
                best_vars = best_init_full
                best_sum = best_center_sum
        except Exception:
            best_vars = best_init_full
            best_sum = best_center_sum
    else:
        best_vars = inits_centers[0]
        best_sum = 0.0
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    return centers, radii, float(best_sum)
