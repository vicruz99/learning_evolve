# sol_000028 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d17cbe8) state=1c5b6a86 sum of radii=2.597459 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_pair_indices(n):
    """Precompute indices for all unique circle pairs."""
    i_idx = []
    j_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            i_idx.append(i)
            j_idx.append(j)
    return np.array(i_idx), np.array(j_idx)

def objective_function(vars, n):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(vars[2*n:])

def constraint_function(vars, n, pair_i, pair_j):
    """Compute inequality constraints: boundaries and non-overlap."""
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    cons = np.concatenate([
        centers[:, 0] - radii,
        1 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1 - centers[:, 1] - radii
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    c_i = centers[pair_i]
    c_j = centers[pair_j]
    r_i = radii[pair_i]
    r_j = radii[pair_j]
    
    dist_sq = np.sum((c_i - c_j)**2, axis=1)
    r_sum = r_i + r_j
    
    cons = np.concatenate([cons, dist_sq - r_sum**2])
    return cons

def generate_initial_config(n, config_type, seed=None):
    """Generate initial center positions and small radii."""
    if seed is not None:
        np.random.seed(seed)
        
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.06)
    
    if config_type == 'grid':
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < n:
                    centers[idx] = [i/4.0 + 0.05, j/4.0 + 0.05]
                    idx += 1
        while idx < n:
            centers[idx] = [np.random.rand(), np.random.rand()]
            idx += 1
            
    elif config_type == 'hex':
        idx = 0
        y = 0.08
        while idx < n and y < 0.92:
            x = 0.08
            while x < 0.92 and idx < n:
                centers[idx] = [x, y]
                idx += 1
                x += 0.16
            y += 0.14
        while idx < n:
            centers[idx] = [np.random.rand(), np.random.rand()]
            idx += 1
            
    elif config_type == 'random':
        centers = np.random.rand(n, 2)
        
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    pair_i, pair_j = get_pair_indices(n)
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
    
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Multiple starts to avoid local optima
    configs = [
        ('grid', 0),
        ('hex', 0),
        ('random', 42),
        ('random', 123),
        ('random', 456)
    ]
    
    for ctype, seed in configs:
        x0 = generate_initial_config(n, ctype, seed)
        
        res = minimize(objective_function, x0, args=(n,), method='SLSQP',
                       bounds=bounds,
                       constraints={'type': 'ineq', 'fun': constraint_function, 'args': (n, pair_i, pair_j)},
                       options={'maxiter': 2000, 'ftol': 1e-12})
                       
        if res.success:
            centers = res.x[:2*n].reshape(n, 2)
            radii = res.x[2*n:]
            
            # Strict validity check
            valid = True
            if np.any(centers[:, 0] - radii < -1e-9) or np.any(centers[:, 0] + radii > 1 + 1e-9) or \
               np.any(centers[:, 1] - radii < -1e-9) or np.any(centers[:, 1] + radii > 1 + 1e-9):
                valid = False
                
            if valid:
                diffs = centers[pair_i] - centers[pair_j]
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                r_sums = radii[pair_i] + radii[pair_j]
                if np.any(dists < r_sums - 1e-9):
                    valid = False
                    
            if valid:
                current_sum = np.sum(radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                    
    return best_centers, best_radii, best_sum
