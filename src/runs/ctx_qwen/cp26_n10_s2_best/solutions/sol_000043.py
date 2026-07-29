# sol_000043 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000016 (state 585439f0) state=031a9d33 sum of radii=1.263376 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def constraints(v):
    """Inequality constraints: boundaries and pairwise non-overlap (squared)."""
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c_boundary = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise constraints: dist^2 >= (r_i + r_j)^2
    i, j = np.triu_indices(N, k=1)
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    dr = r[i] + r[j]
    
    c_pairwise = dx**2 + dy**2 - dr**2
    
    return np.concatenate([c_boundary, c_pairwise])

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_val = -1e9
    best_x = None
    
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])
    
    # Multi-start optimization with hexagonal lattice initialization
    for seed in range(50):
        np.random.seed(seed)
        
        # Generate hexagonal lattice points
        r_base = 0.095
        centers = []
        y = r_base
        row = 0
        while len(centers) < N + 10:
            x_start = r_base + (row % 2) * r_base
            x = x_start
            while x <= 1 - r_base:
                centers.append([x, y])
                x += 2 * r_base
            y += np.sqrt(3) * r_base
            row += 1
            
        centers = np.array(centers[:N])
        # Add controlled perturbation to break symmetry and explore basins
        centers += np.random.uniform(-0.06, 0.06, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        x0 = np.concatenate([centers.flatten(), np.full(N, 0.085)])
        
        # Primary optimization
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                       
        val = -res.fun
        # Accept if feasible and improves best
        if np.min(constraints(res.x)) >= -1e-8 and val > best_val:
            best_val = val
            best_x = res.x.copy()
            
        # Hill-climbing refinement: perturb best found solution and re-optimize
        if best_x is not None:
            x_pert = best_x + np.random.uniform(-0.02, 0.02, best_x.shape)
            x_pert = np.clip(x_pert, lb, ub)
            
            res2 = minimize(objective, x_pert, method='SLSQP', bounds=bounds, constraints=cons,
                            options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            val2 = -res2.fun
            if np.min(constraints(res2.x)) >= -1e-8 and val2 > best_val:
                best_val = val2
                best_x = res2.x.copy()

    centers = best_x[:2*N].reshape(N, 2)
    radii = best_x[2*N:]
    
    # Strict post-processing to guarantee validation passes
    radii = np.maximum(radii, 0.0)
    
    # Enforce boundary constraints strictly
    for i in range(N):
        radii[i] = min(radii[i], centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        
    # Enforce non-overlap strictly with safety margin
    for i in range(N):
        for j in range(i+1, N):
            dist = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
            sum_r = radii[i] + radii[j]
            if dist < sum_r - 1e-12:
                shrink = (sum_r - dist)/2.0 + 1e-8
                radii[i] = max(0.0, radii[i] - shrink)
                radii[j] = max(0.0, radii[j] - shrink)
                
    return centers, radii, float(np.sum(radii))
