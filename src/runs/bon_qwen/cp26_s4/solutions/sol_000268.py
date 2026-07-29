# sol_000268 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f6ad2c92) state=2bbcb3fe sum of radii=2.469064 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars_):
    # Maximize sum of radii by minimizing negative sum
    return -np.sum(vars_[2::3])

def constraint_fun(vars_):
    x = vars_[0::3]
    y = vars_[1::3]
    r = vars_[2::3]
    
    # Boundary constraints: circles must stay inside [0,1]x[0,1]
    c_bound = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Vectorized computation for efficiency
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    r_sum_sq = (r[:, None] + r[None, :])**2
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c_pair = dist_sq[mask] - r_sum_sq[mask]
    
    return np.concatenate([c_bound, c_pair])

def run_packing():
    best_sum = -1.0
    best_c = None
    best_r = None
    
    # Multiple restarts to avoid poor local optima
    for trial in range(3):
        np.random.seed(trial * 1234)
        
        # Initialize with a perturbed hexagonal grid
        centers = []
        for i in range(5):
            y = 0.15 + i * 0.175
            x_start = 0.12 if i % 2 == 0 else 0.22
            for j in range(5):
                if len(centers) < N:
                    centers.append([x_start + j * 0.2, y])
                    
        while len(centers) < N:
            centers.append([0.5, 0.5])
            
        centers = np.array(centers[:N])
        centers += np.random.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        r_init = np.full(N, 0.09)
        
        # Flatten to [x1, y1, r1, x2, y2, r2, ...]
        vars0 = np.zeros(3 * N)
        for i in range(N):
            vars0[3*i] = centers[i, 0]
            vars0[3*i+1] = centers[i, 1]
            vars0[3*i+2] = r_init[i]
            
        bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
        cons = {'type': 'ineq', 'fun': constraint_fun}
        
        res = minimize(objective, vars0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
                       
        curr_sum = np.sum(res.x[2::3])
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_c = np.column_stack((res.x[0::3], res.x[1::3]))
            best_r = res.x[2::3]
            
    # Ensure non-negative radii (solver should maintain this, but safety first)
    if best_r is not None:
        best_r = np.maximum(best_r, 0.0)
        
    # Fallback in extremely rare case of complete failure
    if best_c is None:
        best_c = np.tile([0.5, 0.5], (N, 1))
        best_r = np.full(N, 0.05)
        best_sum = np.sum(best_r)
        
    return best_c, best_r, best_sum
