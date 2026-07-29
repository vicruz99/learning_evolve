# sol_000009 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b4d6f452) state=987046ee sum of radii=2.544371 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def _objective(x):
    radii = x[2*N_CIRCLES:]
    return -np.sum(radii)

def _constraint_fun(x):
    n = N_CIRCLES
    centers = x[:2*n].reshape(n, 2)
    radii = x[2*n:]
    
    num_constraints = 4*n + n*(n-1)//2
    cons = np.empty(num_constraints)
    
    # Boundary constraints: r <= x,y <= 1-r
    cons[:4*n] = np.concatenate([
        centers[:, 0] - radii,
        1 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1 - centers[:, 1] - radii
    ]).ravel()
    
    # Pairwise non-overlap constraints: ||c_i - c_j|| >= r_i + r_j
    idx = 4*n
    for i in range(n):
        diffs = centers[i] - centers[i+1:]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        r_sums = radii[i] + radii[i+1:]
        cons[idx : idx + len(dists)] = dists - r_sums
        idx += len(dists)
        
    return cons

def _get_initial_guess():
    n = N_CIRCLES
    pts = []
    # Generate a hexagonal lattice pattern
    for row in range(20):
        for col in range(20):
            x = col * 0.12 + (row % 2) * 0.06
            y = row * 0.11
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                pts.append([x, y])
    pts = pts[:n]
    
    if len(pts) < n:
        pts = np.random.rand(n, 2) * 0.8 + 0.1
        
    centers = np.array(pts)
    radii = np.full(n, 0.04)  # Start with small radii to ensure initial feasibility
    return np.concatenate([centers.ravel(), radii])

def run_packing():
    np.random.seed(42)
    n = N_CIRCLES
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
    constraints = {'type': 'ineq', 'fun': _constraint_fun}
    
    best_sum = -np.inf
    best_x = None
    
    for trial in range(3):
        x0 = _get_initial_guess()
        if trial > 0:
            # Perturb to avoid identical runs and escape local minima
            pert = np.random.randn(len(x0)) * 0.01
            x0 = x0 + pert
            for i in range(2*n):
                x0[i] = np.clip(x0[i], 0, 1)
            for i in range(2*n, 3*n):
                x0[i] = np.clip(x0[i], 0, 0.5)
                
        res = minimize(_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_x = res.x

    centers = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]
    
    # Final clamping to guarantee boundary constraints hold within numerical precision
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1-r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1-r)
        
    return centers, radii, np.sum(radii)
