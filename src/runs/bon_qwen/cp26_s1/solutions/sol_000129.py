# sol_000129 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 22de7e34) state=631e563b sum of radii=2.236276 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Initial Configuration (Hexagonal Lattice)
    n = 26
    centers = np.zeros((n, 2))
    
    # Distribute circles in staggered rows to mimic hexagonal packing
    # Row sizes: 5, 5, 5, 5, 6 (Total 26)
    row_counts = [5, 5, 5, 5, 6]
    idx = 0
    r_init = 0.095 # Initial estimate
    
    for row_idx, count in enumerate(row_counts):
        y = r_init + row_idx * (r_init * np.sqrt(3))
        # Calculate x spacing
        # Width = 1, count circles -> (count-1) gaps
        # If staggered, x_offset varies
        x_offset = 0.5 if row_idx % 2 == 1 else 0.0 # Slight shift for staggering
        
        # Distribute x centers evenly across [0.1, 0.9] range initially
        step = (1.0 - 2 * r_init) / count if count > 1 else 0
        if count > 0:
            # Adjust start to center the row
            total_width = (count - 1) * step
            start_x = (1.0 - total_width) / 2
            
            for i in range(count):
                centers[idx, 0] = start_x + i * step
                centers[idx, 1] = y
                idx += 1

    # 2. Optimization
    def objective(p_flat):
        # p_flat contains 26*2 = 52 coordinates
        p = p_flat.reshape(-1, 2)
        return -calculate_min_radius(p)

    def calculate_min_radius(p):
        n = p.shape[0]
        min_r = 1.0
        
        # 1. Distance to boundaries
        min_r = np.min([np.min(p[:, 0]), np.min(p[:, 1]), 
                        np.min(1 - p[:, 0]), np.min(1 - p[:, 1])])
        
        # 2. Distance between circles
        if n > 1:
            # Compute pairwise distances
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            # We only need lower triangle (i < j)
            # dists[i, i] is 0, so we filter
            dists = dists[np.triu_indices(n, k=1)]
            if len(dists) > 0:
                min_dist = np.min(dists)
                min_r = min(min_r, min_dist / 2.0)
        
        return min_r

    # Bounds for centers (must be within [0, 1])
    bounds = [(0.0, 1.0) for _ in range(n * 2)]

    # Run optimization
    initial_guess = centers.flatten()
    
    # Using SLSQP for bound-constrained optimization
    result = minimize(objective, initial_guess, method='SLSQP', 
                      bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
    
    final_centers = result.x.reshape(-1, 2)
    final_r = calculate_min_radius(final_centers)
    final_radii = np.full(n, final_r)
    
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
