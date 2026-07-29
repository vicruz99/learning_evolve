# sol_000022 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d17cbe8) state=88f7f0ca sum of radii=0.159394 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(v):
    """Objective function: minimize negative sum of radii."""
    # Radii are located at indices 2, 5, 8, ... in the flat array
    return -np.sum(v[2::3])

def constraint_fun(v):
    """
    Returns a flat array of constraint values. 
    All values must be >= 0 for feasibility.
    """
    # Reshape flat variables into centers (N, 2) and radii (N,)
    centers = v[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = v[2 * N_CIRCLES:]
    
    # 1. Boundary constraints: circle must be inside [0,1]^2
    # Conditions: x >= r, 1-x >= r, y >= r, 1-y >= r
    con_boundary = np.zeros(4 * N_CIRCLES)
    con_boundary[0::4] = centers[:, 0] - radii          # x - r >= 0
    con_boundary[1::4] = 1.0 - centers[:, 0] - radii    # 1 - x - r >= 0
    con_boundary[2::4] = centers[:, 1] - radii          # y - r >= 0
    con_boundary[3::4] = 1.0 - centers[:, 1] - radii    # 1 - y - r >= 0
    
    # 2. Non-overlap constraints: dist(i,j) >= r_i + r_j
    # Vectorized pairwise distance computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Vectorized pairwise radius sums
    r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Extract upper triangle indices (i < j) to avoid duplicates and self-pairs
    tri_idx = np.triu_indices(N_CIRCLES, k=1)
    con_overlap = dists[tri_idx] - r_sums[tri_idx]
    
    # Combine and return all constraints
    return np.concatenate([con_boundary, con_overlap])

def run_packing():
    np.random.seed(42)
    
    # Initial guess: Hexagonal-like grid packing
    # This provides a feasible starting configuration close to optimal density
    v0 = np.zeros(3 * N_CIRCLES)
    count = 0
    y = 0.12
    row = 0
    while count < N_CIRCLES:
        x = 0.12 if row % 2 == 0 else 0.22
        while count < N_CIRCLES and x < 0.95:
            v0[3*count] = x
            v0[3*count+1] = y
            v0[3*count+2] = 0.06  # Initial radius (feasible since spacing > 0.2)
            count += 1
            x += 0.20
        y += 0.17
        row += 1
        
    # Add small random perturbation to break symmetry and aid optimization
    v0 += np.random.randn(*v0.shape) * 0.005
    
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    
    # Constraint specification for SLSQP (inequality: g(x) >= 0)
    cons = ({'type': 'ineq', 'fun': constraint_fun})
    
    # Run SLSQP optimizer
    res = minimize(objective, v0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
    
    # Extract results
    centers_opt = res.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii_opt = res.x[2 * N_CIRCLES:]
    
    # Post-processing: ensure strict feasibility for validation tolerance
    # 1. Clip radii to boundary limits
    for i in range(N_CIRCLES):
        cx, cy = centers_opt[i]
        max_r_boundary = min(cx, 1.0 - cx, cy, 1.0 - cy)
        if radii_opt[i] > max_r_boundary:
            radii_opt[i] = max_r_boundary
            
    # 2. Conservatively scale down overlapping pairs if numerical error occurred
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            dist = np.sqrt(np.sum((centers_opt[i] - centers_opt[j]) ** 2))
            sum_r = radii_opt[i] + radii_opt[j]
            if sum_r > dist + 1e-12:
                scale = dist / sum_r * 0.999999
                radii_opt[i] *= scale
                radii_opt[j] *= scale
                
    sum_radii = float(np.sum(radii_opt))
    return centers_opt, radii_opt, sum_radii
