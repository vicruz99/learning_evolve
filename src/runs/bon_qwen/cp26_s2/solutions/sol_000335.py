# sol_000335 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d755ba05) state=8ce4b394 sum of radii=2.629868 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(x):
    # Minimize negative sum of radii -> Maximizes sum of radii
    return -np.sum(x[2*N:])

def boundary_constraints(x):
    # Ensures circles are within [0,1]x[0,1]
    c = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    con = np.empty(4*N)
    con[:N] = c[:, 0] - r          # x - r >= 0
    con[N:2*N] = 1.0 - c[:, 0] - r # 1 - x - r >= 0
    con[2*N:3*N] = c[:, 1] - r     # y - r >= 0
    con[3*N:] = 1.0 - c[:, 1] - r  # 1 - y - r >= 0
    return con

def separation_constraints(x):
    # Ensures no overlaps: ||c_i - c_j||^2 >= (r_i + r_j)^2
    c = x[:2*N].reshape(N, 2)
    r = x[2*N:]
    
    # Compute all pairwise squared distances and squared radius sums
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    rad_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    # Extract strictly upper triangle (i < j) to avoid duplicates and self-comparisons
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return dist_sq[mask] - rad_sum_sq[mask]

def run_packing():
    np.random.seed(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Prepare initial configurations
    initial_configs = []
    
    # 1. Hexagonal packing (5 rows of 5, alternating shift) + 1 extra
    c1 = np.zeros((N, 2))
    r1 = np.full(N, 0.095)
    idx = 0
    row_h = np.sqrt(3) * 0.095
    col_w = 2 * 0.095
    for row in range(5):
        y = 0.1 + row * row_h
        x_start = 0.1 + (row % 2) * (col_w / 2)
        for col in range(5):
            c1[idx] = [x_start + col * col_w, y]
            idx += 1
    c1[25] = [0.05, 0.5]
    initial_configs.append((c1, r1))
    
    # 2. Square grid packing (5x5) + 1 extra
    c2 = np.zeros((N, 2))
    r2 = np.full(N, 0.09)
    idx = 0
    for i in range(5):
        for j in range(5):
            c2[idx] = [0.1 + i * 0.1, 0.1 + j * 0.1]
            idx += 1
    c2[25] = [0.55, 0.55]
    initial_configs.append((c2, r2))
    
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    constraints = [
        {'type': 'ineq', 'fun': boundary_constraints},
        {'type': 'ineq', 'fun': separation_constraints}
    ]
    
    # Optimize from multiple perturbed starts
    for c_init, r_init in initial_configs:
        for _ in range(4):
            # Perturb initial guess
            cp = c_init + np.random.normal(0, 0.008, c_init.shape)
            rp = r_init + np.random.normal(0, 0.003, N)
            cp = np.clip(cp, 0.02, 0.98)
            rp = np.clip(rp, 0.05, 0.25)
            
            x0 = np.concatenate([cp.flatten(), rp])
            
            res = minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False}
            )
            
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = res.x[:2*N].reshape(N, 2)
                best_radii = res.x[2*N:]
                
    # Safety clamping
    best_radii = np.maximum(best_radii, 0.0)
    best_centers = np.clip(best_centers, 0.0, 1.0)
    
    return best_centers, best_radii, np.sum(best_radii)
