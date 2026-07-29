# sol_000019 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfb3fe63) state=7a647909 sum of radii=1.484501 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_min_dist(centers):
    """Compute the minimum distance between any pair of points and to the square boundaries."""
    n = centers.shape[0]
    if n == 0:
        return 0.0
        
    # Distance to boundaries: min(x, 1-x, y, 1-y)
    bound_dists = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                             np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    min_bound = np.min(bound_dists)
    
    # Pairwise distances using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_bound, min_pair)

def objective(vars):
    """Objective function to minimize: negative of the minimum distance."""
    centers = np.array(vars).reshape(26, 2)
    centers = np.clip(centers, 0.0, 1.0)
    return -compute_min_dist(centers)

def run_packing():
    n = 26
    best_obj = np.inf
    best_vars = None
    
    # Initial configurations
    inits = []
    
    # 1. 5x5 Grid + 1 center circle
    g = np.linspace(0.1, 0.9, 5)
    grid = np.zeros((25, 2))
    k = 0
    for i in range(5):
        for j in range(5):
            grid[k] = [g[i], g[j]]
            k += 1
    grid = np.vstack([grid, [0.5, 0.5]])
    inits.append(grid.flatten())
    
    # 2. Hexagonal-like staggered layout (6-5-6-5-4)
    hex_pat = []
    ys = [0.10, 0.27, 0.44, 0.61, 0.78]
    for idx, y in enumerate(ys):
        n_circles = 6 if idx % 2 == 0 else 5
        xs = np.linspace(0.10, 0.90, n_circles)
        if idx % 2 == 1:
            xs = np.linspace(0.15, 0.85, n_circles)
        for x in xs:
            hex_pat.append([x, y])
    hex_pat = np.array(hex_pat[:26])
    inits.append(hex_pat.flatten())
    
    # 3. Random starts for robustness
    np.random.seed(42)
    for _ in range(4):
        inits.append((np.random.rand(n, 2) * 0.8 + 0.1).flatten())
        
    # Optimize from each start
    for init in inits:
        res = minimize(objective, init, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.fun < best_obj:
            best_obj = res.fun
            best_vars = res.x
            
    centers = best_vars.reshape(n, 2)
    centers = np.clip(centers, 0.0, 1.0)
    
    min_d = compute_min_dist(centers)
    radii = np.full(n, min_d / 2.0)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
