# sol_000223 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state df9a626f) state=4db74773 sum of radii=2.482305 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_max_radii(centers):
    """
    Computes the maximum possible radius for each circle given fixed centers.
    Radius is constrained by distance to boundaries and distance to other circles.
    """
    n = centers.shape[0]
    x, y = centers[:, 0], centers[:, 1]
    
    # Distance to the four boundaries
    dist_to_boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    # Pairwise distances between centers using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)  # Ignore self-distance
    
    # Min distance to any other circle center, divided by 2
    dist_to_neighbors = np.min(dists, axis=1) / 2.0
    
    # The radius is constrained by the tighter of the two limits
    return np.minimum(dist_to_boundary, dist_to_neighbors)

def objective(centers_flat):
    """
    Objective function to minimize: negative sum of maximum possible radii.
    """
    centers = centers_flat.reshape(26, 2)
    radii = get_max_radii(centers)
    return -np.sum(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    
    # Bounds for all 26 centers (x, y must be in [0, 1])
    bounds = [(0.0, 1.0)] * (2 * n)
    
    # Generate diverse initial configurations to avoid poor local minima
    initial_configs = []
    
    # 1. Random distributions
    for _ in range(4):
        initial_configs.append(np.random.uniform(0.1, 0.9, (n, 2)))
        
    # 2. Structured Grid (5x5) with perturbation
    grid = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < n:
                grid[idx] = [0.1 + j * 0.2, 0.1 + i * 0.2]
                idx += 1
    # Place 26th circle in the center gap
    if idx < n:
        grid[idx] = [0.5, 0.5]
    initial_configs.append(grid + np.random.normal(0, 0.03, (n, 2)))
    
    # 3. Hexagonal lattice pattern
    hex_c = []
    r_init = 0.09
    y_pos = 0.1
    row = 0
    while len(hex_c) < n:
        x_pos = 0.1
        shift = r_init if row % 2 == 1 else 0.0
        while x_pos + r_init <= 1.0 and len(hex_c) < n:
            hex_c.append([x_pos + shift, y_pos])
            x_pos += 2.0 * r_init
        y_pos += r_init * np.sqrt(3)
        row += 1
    initial_configs.append(np.array(hex_c[:n]))

    # Optimization loop over all initial guesses
    for init_centers in initial_configs:
        # Clamp initial centers strictly inside bounds to avoid optimizer warnings
        init_flat = np.clip(init_centers.flatten(), 1e-6, 1.0 - 1e-6)
        
        res = minimize(objective, init_flat, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 15000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        curr_sum = -res.fun
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = res.x.reshape(n, 2)
            
    # Compute final radii for the best configuration
    final_radii = get_max_radii(best_centers)
    
    return best_centers, final_radii, np.sum(final_radii)
