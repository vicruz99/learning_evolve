# sol_000375 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 469c683e) state=8883139d sum of radii=2.598563 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(v):
    # Minimize negative sum of radii to maximize sum
    return -np.sum(v[52:])

def compute_constraints(v):
    n = 26
    centers = v[:52].reshape(n, 2)
    radii = v[52:]
    
    cons = []
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    # x - r >= 0
    cons.append(centers[:, 0] - radii)
    # 1 - x - r >= 0
    cons.append(1.0 - centers[:, 0] - radii)
    # y - r >= 0
    cons.append(centers[:, 1] - radii)
    # 1 - y - r >= 0
    cons.append(1.0 - centers[:, 1] - radii)
    
    # Overlap constraints: dist(i,j) - (r_i + r_j) >= 0
    # Compute pairwise distances using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Lower triangular indices (excluding diagonal)
    i_idx, j_idx = np.tril_indices(n, -1)
    overlap_con = dists[i_idx, j_idx] - (radii[i_idx] + radii[j_idx])
    cons.append(overlap_con)
    
    return np.concatenate(cons)

def run_packing():
    # Initial configuration: Hexagonal lattice
    # Rows configuration for 26 circles
    rows = [4, 5, 6, 5, 6]
    r_init = 0.085  # Start slightly smaller to give optimizer room
    dy = np.sqrt(3) * r_init
    y_base = 0.5
    
    centers = []
    radii = [r_init] * 26
    
    for row_idx, n_circles in enumerate(rows):
        y = y_base + (row_idx - 2) * dy
        x_start = 0.5 - (n_circles - 1) * r_init
        for j in range(n_circles):
            x = x_start + j * 2 * r_init
            centers.append([x, y])
            
    centers = np.array(centers)
    radii = np.array(radii)
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Variable bounds: x, y in [0, 1], r in [1e-5, 0.5]
    bounds = [(0, 1)] * 52 + [(1e-5, 0.5)] * 26
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # Optimize
    res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds, 
                   constraints=cons, options={'ftol': 1e-9, 'maxiter': 3000, 'disp': False})
    
    final_centers = res.x[:52].reshape(26, 2)
    final_radii = res.x[52:]
    
    # Ensure non-negative radii
    final_radii = np.maximum(final_radii, 0.0)
    
    # Strictly enforce boundaries
    final_centers[:, 0] = np.clip(final_centers[:, 0], final_radii, 1.0 - final_radii)
    final_centers[:, 1] = np.clip(final_centers[:, 1], final_radii, 1.0 - final_radii)
    
    return final_centers, final_radii, np.sum(final_radii)
