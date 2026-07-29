# sol_000263 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8d1f387b) state=2426115b sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def compute_optimal_radii(centers):
    """
    Solve the LP to find maximum sum of radii for fixed centers.
    Returns (radii_array, sum_of_radii).
    """
    n = centers.shape[0]
    
    # Distance to boundaries: r_i <= min(x, 1-x, y, 1-y)
    b_i = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                     np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d_ij = np.sqrt(np.sum(diff**2, axis=2))
    
    # LP: max sum(r_i)  <=>  min -sum(r_i)
    c = np.ones(n)
    
    A_ub_rows = []
    b_ub_vals = []
    
    # Constraints: r_i + r_j <= d_ij
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(d_ij[i, j])
            
    # Constraints: r_i <= b_i
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(b_i[i])
        
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        # Fallback for older scipy versions
        res = linprog(-c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        
    if res.success:
        return res.x, -res.fun
    return np.zeros(n), 0.0

def packing_objective(centers_flat):
    """
    Objective function for center optimization.
    Returns negative sum of radii (to minimize).
    Includes heavy penalty if centers violate square bounds.
    """
    centers = centers_flat.reshape(-1, 2)
    
    # Boundary penalty to keep optimizer in [0,1]^2
    violation = np.sum(np.maximum(0.0, -centers)**2 + np.maximum(0.0, centers - 1.0)**2)
    if violation > 1e-6:
        return 1e6 + violation
        
    _, val = compute_optimal_radii(centers)
    return -val

def run_packing() -> tuple:
    np.random.seed(42)
    n = 26
    best_centers = np.zeros((n, 2))
    best_radii = np.zeros(n)
    best_sum = -1.0
    
    # Run multiple optimization trials from different perturbations
    for trial in range(4):
        # Initialize on a hexagonal-like grid
        pts = []
        x_step = 0.20
        y_step = x_step * np.sqrt(3) / 2.0
        for i in range(6):
            for j in range(5):
                x = 0.08 + j * x_step
                y = 0.08 + i * y_step
                if i % 2 == 1:
                    x += x_step / 2.0
                pts.append([x, y])
                if len(pts) == n:
                    break
            if len(pts) == n:
                break
                
        init_centers = np.array(pts)
        # Add controlled random jitter
        jitter = np.random.normal(0, 0.025, init_centers.shape)
        init_centers += jitter
        init_centers = np.clip(init_centers, 0.05, 0.95)
        
        # Optimize centers
        res = minimize(packing_objective, init_centers.flatten(),
                       method='Nelder-Mead',
                       options={'maxiter': 250, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
                       
        if res.success:
            opt_centers = res.x.reshape(-1, 2)
            opt_centers = np.clip(opt_centers, 1e-5, 1.0 - 1e-5)
            opt_radii, opt_sum = compute_optimal_radii(opt_centers)
            
            if opt_sum > best_sum:
                best_sum = opt_sum
                best_centers = opt_centers
                best_radii = opt_radii
                
    return best_centers, best_radii, float(best_sum)
