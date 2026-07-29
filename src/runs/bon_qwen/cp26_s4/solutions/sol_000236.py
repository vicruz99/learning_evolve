# sol_000236 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1140c965) state=96bc6808 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(vars):
    """Objective: maximize radius r (minimize -r)"""
    return -vars[-1]

def constraint_func(vars):
    """Constraints: boundary containment and non-overlap (squared distances)"""
    n = 26
    c = vars[:2*n].reshape(n, 2)
    r = vars[-1]
    cons = []
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons.append(c[:, 0] - r)
    cons.append(1.0 - c[:, 0] - r)
    cons.append(c[:, 1] - r)
    cons.append(1.0 - c[:, 1] - r)
    
    # Overlap constraints: dist_sq >= (2r)^2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Extract upper triangle to avoid duplicate constraints
    upper_sq = np.triu(dists_sq, k=1)
    upper_sq = upper_sq[upper_sq < np.inf]
    cons.append(upper_sq - 4*r**2)
    
    return np.concatenate(cons)

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    idx = 0
    
    # Initialize with a 5x5 grid + 1 center circle
    for i in range(5):
        for j in range(5):
            centers[idx] = [i*0.2 + 0.1, j*0.2 + 0.1]
            idx += 1
    centers[idx] = [0.5, 0.5]
    
    # Deterministic perturbation to break symmetry and aid convergence
    np.random.seed(42)
    centers += np.random.uniform(-0.005, 0.005, size=centers.shape)
    centers = np.clip(centers, 0.01, 0.99)
    
    # Initial variable vector: [x1, y1, ..., x26, y26, r]
    x0 = np.concatenate([centers.ravel(), [0.08]])
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 1.0)]
    
    # Run SLSQP optimizer
    res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, 
                   constraints={'type': 'ineq', 'fun': constraint_func},
                   options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                   
    best = res.x if res.success else x0
    centers = best[:2*n].reshape(n, 2)
    r = best[-1]
    
    # Post-processing: strictly enforce constraints with numerical tolerance
    # Boundary clamping
    r = min(r, np.min([np.min(centers[:,0]), np.min(1- centers[:,0]), 
                       np.min(centers[:,1]), np.min(1- centers[:,1])]) - 1e-9)
                       
    # Overlap clamping
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_dist_sq = np.min(dists_sq[dists_sq < np.inf])
    r = min(r, np.sqrt(min_dist_sq) / 2.0 - 1e-9)
    
    radii = np.full(n, r)
    return centers, radii, np.sum(radii)
