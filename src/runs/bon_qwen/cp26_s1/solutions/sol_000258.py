# sol_000258 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3be09fa9) state=62aa0035 sum of radii=2.601045 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(v, N):
    """Maximize sum of radii => minimize negative sum."""
    return -np.sum(v[2*N:])

def constraint_func(v, N):
    """
    Returns array of constraint values. All must be >= 0.
    Constraints:
    1. Boundary: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    2. Non-overlap: dist(i,j) - r_i - r_j >= 0
    """
    centers = v[:2*N].reshape(N, 2)
    radii = v[2*N:]
    
    con = []
    # Boundary constraints
    con.extend(centers[:, 0] - radii)
    con.extend(1 - centers[:, 0] - radii)
    con.extend(centers[:, 1] - radii)
    con.extend(1 - centers[:, 1] - radii)
    
    # Pairwise non-overlap constraints (vectorized)
    # Compute all pairwise differences
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Extract upper triangle indices to avoid duplicates and self-comparison
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    con.extend(dists[mask] - r_sums[mask])
    
    return np.array(con)

def run_packing():
    N = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    
    # Optimization settings
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraint_func, 'args': (N,)}
    
    # Multi-start optimization
    for _ in range(5):
        # Generate hexagonal grid initialization
        pts = []
        r_init = 0.075
        dx = 2.0 * r_init
        dy = np.sqrt(3.0) * r_init
        
        for i in range(8):
            for j in range(8):
                x = j * dx + (dx/2.0 if i % 2 == 1 else 0.0)
                y = i * dy
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append([x, y])
        
        pts = np.array(pts[:N])
        # Add small random perturbation
        pts += np.random.uniform(-0.005, 0.005, pts.shape)
        pts = np.clip(pts, 0.01, 0.99)
        
        radii = np.full(N, r_init)
        x0 = np.concatenate([pts.ravel(), radii])
        
        res = minimize(objective_func, x0, args=(N,), method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
        
        current_sum = -res.fun
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = res.x[:2*N].reshape(N, 2)
            best_radii = res.x[2*N:]
            
    return best_centers, best_radii, best_sum
