# sol_000187 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5c6e3651) state=b58ef9a6 sum of radii=2.066639 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def run_packing():
    np.random.seed(42)
    N = 26
    best_min_d = 0
    best_centers = None
    beta = 250.0  # Controls sharpness of smooth min approximation
    
    def objective(x):
        c = x.reshape(-1, 2)
        # Minimum distance to boundaries
        d_b = np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), 
                         np.minimum(c[:, 1], 1 - c[:, 1]))
        # Minimum pairwise distance
        d_p = pdist(c)
        
        # Combine all constraints
        all_d = np.concatenate([d_b, d_p])
        min_d = np.min(all_d)
        
        # Smooth approximation of min: -log(sum(exp(-beta*(d - min_d))))/beta + min_d
        # This avoids overflow and provides a smooth landscape for optimization
        s_min = min_d - (1.0/beta) * np.log(np.sum(np.exp(-beta * (all_d - min_d))))
        return -s_min

    # Each coordinate is bounded in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Base hexagonal grid initialization
    r_init = 0.08
    centers_base = []
    row_counts = [6, 5, 6, 5, 4]
    h = r_init * np.sqrt(3)
    y = 0.1
    
    for count in row_counts:
        x_start = 0.5 - (count - 1) * r_init
        for j in range(count):
            centers_base.append([x_start + j * 2 * r_init, y])
        y += h
    x_base = np.array(centers_base).flatten()
    
    # Run multiple trials with slight perturbations to find global optimum
    for trial in range(6):
        x0 = x_base + np.random.normal(0, 0.004, size=len(x_base))
        x0 = np.clip(x0, 0.01, 0.99)
        
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 4000, 'ftol': 1e-12})
        
        c = res.x.reshape(-1, 2)
        
        # Compute exact minimum distance for validation and radius calculation
        md_boundary = min(np.min(c[:, 0]), np.min(1 - c[:, 0]), 
                          np.min(c[:, 1]), np.min(1 - c[:, 1]))
        md_pairwise = np.min(pdist(c))
        md = min(md_boundary, md_pairwise)
        
        if md > best_min_d:
            best_min_d = md
            best_centers = c.copy()
            
    # Final radius is half the minimum clearance
    final_r = best_min_d / 2
    radii = np.full(N, final_r)
    
    # Ensure strict boundary compliance
    best_centers[:, 0] = np.clip(best_centers[:, 0], final_r, 1 - final_r)
    best_centers[:, 1] = np.clip(best_centers[:, 1], final_r, 1 - final_r)
    
    return best_centers, radii, np.sum(radii)
