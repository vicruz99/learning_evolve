import numpy as np
from scipy.optimize import minimize

def _objective(vars):
    return -np.sum(vars[52:])

def _constraints_overlap(vars):
    cs = vars[:52].reshape(26, 2)
    rs = vars[52:]
    # Vectorized pairwise squared distances
    diff = cs[:, np.newaxis, :] - cs[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    # Constraint: dist_sq >= (r_i + r_j)^2  =>  dist_sq - (r_i+r_j)^2 >= 0
    constraint_vals = dist_sq - r_sum**2
    return np.triu(constraint_vals, k=1).flatten()

def _constraints_boundary(vars):
    cs = vars[:52].reshape(26, 2)
    rs = vars[52:]
    b = np.empty((26, 4))
    b[:, 0] = cs[:, 0] - rs
    b[:, 1] = 1.0 - cs[:, 0] - rs
    b[:, 2] = cs[:, 1] - rs
    b[:, 3] = 1.0 - cs[:, 1] - rs
    return b.flatten()

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05)
    
    # Hexagonal-inspired layout: 5, 4, 5, 4, 5, 3 circles per row
    row_cfg = [5, 4, 5, 4, 5, 3]
    idx = 0
    for r_idx, cnt in enumerate(row_cfg):
        y = 0.15 + r_idx * 0.14
        x_start = 0.15 if r_idx % 2 == 0 else 0.23
        for c in range(cnt):
            if idx >= n: break
            centers[idx] = [x_start + c * 0.15, y]
            idx += 1
            
    x0 = np.concatenate([centers.flatten(), radii])
    
    cons = [
        {'type': 'ineq', 'fun': _constraints_overlap},
        {'type': 'ineq', 'fun': _constraints_boundary}
    ]
    
    bounds = [(0.0, 1.0)] * 52 + [(0.0, 0.5)] * 26
    
    # Run SLSQP optimization
    res = minimize(_objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 3000, 'ftol': 1e-10})
                   
    best_vars = res.x
    final_centers = best_vars[:52].reshape(26, 2)
    final_radii = best_vars[52:]
    
    # Ensure numerical validity
    final_radii = np.maximum(final_radii, 0.0)
    final_centers = np.clip(final_centers, 0.0, 1.0)
    
    return final_centers, final_radii, np.sum(final_radii)