# sol_000121 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 69804dab) state=c9f4482e sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_objective(x):
    """Objective function: negative sum of radii"""
    return -np.sum(x[2*N:])

def compute_constraints(x):
    """Inequality constraints: boundary and non-overlap"""
    c = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    # Boundary constraints: c >= r and 1-c >= r
    b = np.zeros(4*N)
    b[0::4] = c[:, 0] - r
    b[1::4] = 1.0 - c[:, 0] - r
    b[2::4] = c[:, 1] - r
    b[3::4] = 1.0 - c[:, 1] - r
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    sum_r_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    # Only upper triangle to avoid duplicates
    triu_idx = np.triu_indices(N, k=1)
    overlap_cons = dist_sq[triu_idx] - sum_r_sq[triu_idx]
    
    return np.concatenate([b, overlap_cons])

def run_packing():
    # 1. Initialize in a hexagonal pattern (dense baseline)
    centers = np.zeros((N, 2))
    radii = np.full(N, 0.09)
    
    idx = 0
    for row in range(6):
        y = 0.09 + row * 0.09 * np.sqrt(3)
        cols = 5 if row < 5 else 1
        offset = 0.09 if row % 2 != 0 else 0.0
        for c in range(cols):
            if idx >= N: break
            x = 0.09 + c * 0.18 + offset
            centers[idx] = [x, y]
            idx += 1
            
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds for variables
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    
    # Constraint definition
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # 2. Optimize
    res = minimize(compute_objective, x0, method='SLSQP', 
                   constraints=cons, bounds=bounds, 
                   options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                   
    final_centers = res.x[:2*N].reshape(N, 2)
    final_radii = res.x[2*N:]
    
    # 3. Post-process for strict feasibility
    final_radii = np.maximum(final_radii, 1e-9)
    final_centers[:, 0] = np.clip(final_centers[:, 0], final_radii, 1.0 - final_radii)
    final_centers[:, 1] = np.clip(final_centers[:, 1], final_radii, 1.0 - final_radii)
    
    return final_centers, final_radii, float(np.sum(final_radii))
