# sol_000003 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d5d6e849) state=f506e26f sum of radii=1.040000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def bound_constraint(v, i, N, type_idx):
    """
    Constraint function for boundary conditions.
    type_idx: 0 -> x >= r, 1 -> 1-x >= r, 2 -> y >= r, 3 -> 1-y >= r
    """
    r_idx = 3 * N + i
    if type_idx == 0:
        return v[3 * i] - v[r_idx]
    elif type_idx == 1:
        return 1.0 - v[3 * i] - v[r_idx]
    elif type_idx == 2:
        return v[3 * i + 1] - v[r_idx]
    else:
        return 1.0 - v[3 * i + 1] - v[r_idx]

def overlap_constraint(v, i, j, N):
    """
    Constraint function for non-overlap between circle i and j.
    """
    r_i = 3 * N + i
    r_j = 3 * N + j
    dx = v[3 * i] - v[3 * j]
    dy = v[3 * i + 1] - v[3 * j + 1]
    dist = np.sqrt(dx * dx + dy * dy)
    return dist - (v[r_i] + v[r_j])

def run_packing():
    N = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Initial configuration: Hexagonal grid
    centers = np.zeros((N, 2))
    radii = np.ones(N) * 0.04
    row, col, idx = 0, 0, 0
    while idx < N:
        y = 0.08 + row * 0.15
        x = 0.08 + col * 0.15 + (0.075 if row % 2 == 1 else 0.0)
        centers[idx, 0] = np.clip(x, 0.05, 0.95)
        centers[idx, 1] = np.clip(y, 0.05, 0.95)
        idx += 1
        col += 1
        if col >= 6:
            col = 0
            row += 1
            
    x0 = np.concatenate([centers.ravel(), radii])
    
    bounds = []
    for _ in range(N):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    cons = []
    for i in range(N):
        for t in range(4):
            cons.append({'type': 'ineq', 'fun': bound_constraint, 'args': (i, N, t)})
            
    for i in range(N):
        for j in range(i + 1, N):
            cons.append({'type': 'ineq', 'fun': overlap_constraint, 'args': (i, j, N)})
            
    def objective(v):
        return -np.sum(v[3 * N:])
        
    np.random.seed(42)
    for trial in range(5):
        if trial > 0:
            jitter = np.random.uniform(-0.02, 0.02, size=x0.shape)
            curr_x0 = x0 + jitter
            curr_x0[:3*N] = np.clip(curr_x0[:3*N], 0.05, 0.95)
            curr_x0[3*N:] = 0.04
        else:
            curr_x0 = x0.copy()
            
        try:
            res = minimize(objective, curr_x0, bounds=bounds, constraints=cons,
                           method='SLSQP', options={'maxiter': 5000, 'ftol': 1e-10})
            if res.success or res.nit > 2000:
                curr_sum = np.sum(res.x[3*N:])
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = res.x[:3*N].reshape(N, 2)
                    best_radii = res.x[3*N:]
        except Exception:
            continue
            
    if best_centers is None:
        best_centers = centers
        best_radii = radii
        best_sum = np.sum(radii)
        
    return best_centers, best_radii, best_sum
