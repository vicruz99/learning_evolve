# sol_000259 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a15173c5) state=5c1bd177 sum of radii=2.594948 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_objective(var):
    """Negative sum of radii (for minimization)"""
    return -np.sum(var[2 * N_CIRCLES:])

def compute_constraints(var):
    """Returns array of constraint values that must be >= 0"""
    n = N_CIRCLES
    centers = var[:2 * n].reshape((n, 2))
    radii = var[2 * n:]
    con = []
    
    # Boundary constraints: r <= x <= 1-r and r <= y <= 1-r
    for i in range(n):
        con.append(centers[i, 0] - radii[i])
        con.append(centers[i, 1] - radii[i])
        con.append(1.0 - centers[i, 0] - radii[i])
        con.append(1.0 - centers[i, 1] - radii[i])
        
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx * dx + dy * dy
            r_sum = radii[i] + radii[j]
            con.append(dist_sq - r_sum ** 2)
            
    return np.array(con)

def get_initial_config(r_init, noise_level=0.0):
    """Generates a hexagonal lattice initialization with optional noise"""
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    radii = np.full(n, r_init)
    idx = 0
    row = 0
    
    while idx < n:
        y = r_init + row * r_init * np.sqrt(3)
        if y + r_init > 0.99:
            break
            
        x_start = r_init if row % 2 == 0 else r_init * 1.5
        col = 0
        while idx < n:
            x = x_start + col * 2 * r_init
            if x + r_init > 0.99:
                break
            centers[idx, 0] = x
            centers[idx, 1] = y
            idx += 1
            col += 1
        row += 1
        
    # Fill any remaining circles if lattice didn't cover all (edge case)
    while idx < n:
        centers[idx, 0] = 0.5
        centers[idx, 1] = 0.5
        idx += 1
        
    # Apply perturbation to help escape symmetric local minima
    if noise_level > 0:
        centers += np.random.uniform(-noise_level, noise_level, centers.shape)
        centers = np.clip(centers, r_init + 1e-4, 1 - r_init - 1e-4)
        
    return np.concatenate([centers.ravel(), radii])

def run_packing():
    """
    Optimizes circle packing to maximize sum of radii.
    Returns (centers, radii, sum_radii)
    """
    best_neg_obj = float('inf')
    best_var = None
    
    # Multiple restarts with different initial radii and perturbations
    start_radii = [0.05, 0.06, 0.07, 0.08]
    
    for r_start in start_radii:
        for noise in [0.0, 0.02, 0.05]:
            x0 = get_initial_config(r_start, noise_level=noise)
            
            bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(1e-4, 0.5)] * N_CIRCLES
            cons = {'type': 'ineq', 'fun': compute_constraints}
            
            try:
                res = minimize(
                    compute_objective, 
                    x0, 
                    method='SLSQP', 
                    bounds=bounds, 
                    constraints=cons, 
                    options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
                )
                
                if np.isfinite(res.fun) and res.fun < best_neg_obj:
                    best_neg_obj = res.fun
                    best_var = res.x.copy()
            except Exception:
                continue
                
    if best_var is None:
        # Fallback to simple hex packing if optimization fails
        best_var = get_initial_config(0.05)
        
    centers = best_var[:2 * N_CIRCLES].reshape((N_CIRCLES, 2))
    radii = best_var[2 * N_CIRCLES:]
    
    # Clamp tiny numerical violations to satisfy validator tolerance
    radii = np.clip(radii, 1e-6, None)
    
    return centers, radii, -best_neg_obj
