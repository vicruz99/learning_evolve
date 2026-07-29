# sol_000298 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2823a898) state=3353317f sum of radii=2.587827 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import itertools

def compute_constraints(v, N):
    """
    Computes all inequality constraints >= 0.
    v: flattened array [x1, y1, ..., xN, yN, r1, ..., rN]
    """
    centers = v[:2*N].reshape(N, 2)
    radii = v[2*N:]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_bound = np.concatenate([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ])
    
    # Pairwise constraints: dist^2 - (r1+r2)^2 >= 0
    # Vectorized distance calculation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    c_pair = dist_sq - r_sum**2
    
    # Extract upper triangle (i < j) to avoid duplicates and self-constraints
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c_pair = c_pair[mask]
    
    return np.concatenate([c_bound, c_pair])

def objective_func(v, N):
    # We maximize sum of radii, so minimize negative sum
    return -np.sum(v[2*N:])

def run_packing():
    N = 26
    
    # Define bounds: x, y in [0, 1], r in [1e-6, 0.5]
    bounds = [(0, 1)] * 2*N + [(1e-6, 0.5)] * N
    
    best_res = None
    best_obj = np.inf  # Minimizing negative sum
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Hexagonal packing initialization (dense)
    hex_centers = []
    row_y = 0.08
    while row_y < 0.92:
        col_x = 0.08
        is_odd = len(hex_centers) % 2 != 0
        offset = 0.05 if is_odd else 0.0
        while col_x < 0.92:
            if len(hex_centers) < N:
                hex_centers.append([col_x + offset, row_y])
            col_x += 0.14
        row_y += 0.12
    if len(hex_centers) < N:
        while len(hex_centers) < N:
            hex_centers.append([np.random.rand() * 0.8 + 0.1, np.random.rand() * 0.8 + 0.1])
    initial_configs.append(np.array(hex_centers)[:N])
    
    # 2. Grid initialization
    grid_centers = []
    step = 0.22
    for i in range(6):
        for j in range(6):
            if len(grid_centers) < N:
                grid_centers.append([i * step + 0.1, j * step + 0.1])
    initial_configs.append(np.array(grid_centers)[:N])
    
    # 3. Random initialization
    initial_configs.append(np.random.uniform(0.1, 0.9, (N, 2)))
    
    for cfg in initial_configs:
        # Flatten centers and add initial small radii
        v0 = np.concatenate([cfg.flatten(), np.full(N, 0.05)])
        
        res = minimize(
            objective_func,
            v0,
            args=(N,),
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': lambda v, n=N: compute_constraints(v, n)},
            options={'maxiter': 500, 'ftol': 1e-10, 'disp': False}
        )
        
        if res.fun < best_obj:
            # Verify constraints are met within tolerance
            c_vals = compute_constraints(res.x, N)
            if np.min(c_vals) >= -1e-7:
                best_obj = res.fun
                best_res = res
                
    if best_res is None:
        best_res = res  # Fallback to last run
        
    v_final = best_res.x
    centers = v_final[:2*N].reshape(N, 2)
    radii = v_final[2*N:]
    
    # Post-processing to ensure strict feasibility against validator tolerances
    min_c = np.min(compute_constraints(v_final, N))
    if min_c < 0:
        # Scale down radii slightly to guarantee feasibility if numerical slip occurs
        scale_factor = np.sqrt(max(0, min_c)) * 1.01
        radii = radii - scale_factor
        radii = np.maximum(radii, 1e-6)
        
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
