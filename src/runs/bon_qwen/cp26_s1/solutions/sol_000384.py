# sol_000384 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9347ba90) state=2409e304 sum of radii=1.398676 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _packing_objective(params, n):
    """
    Objective function for L-BFGS-B optimization.
    Maximizes radius r while penalizing overlaps and boundary violations.
    """
    c = params[:2*n].reshape(n, 2)
    r = params[2*n]
    
    if r <= 0:
        return 1e9
        
    # Compute pairwise distances efficiently
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Ignore self-distances by setting diagonal to a large value
    np.fill_diagonal(dist, 1e6)
    
    # Overlap penalty: circles must be at least 2r apart
    overlap = np.maximum(0, 2*r - dist)
    pen_overlap = np.sum(overlap**2)
    
    # Boundary penalty: circles must stay within [0,1]x[0,1]
    pen_bound = np.sum(np.maximum(0, r - c[:, 0])**2)
    pen_bound += np.sum(np.maximum(0, c[:, 0] - (1 - r))**2)
    pen_bound += np.sum(np.maximum(0, r - c[:, 1])**2)
    pen_bound += np.sum(np.maximum(0, c[:, 1] - (1 - r))**2)
    
    # Objective: maximize r, minimize penalties
    return -r + 10000.0 * (pen_overlap + pen_bound)

def run_packing():
    n = 26
    
    # 1. Initialize centers in a high-density hexagonal grid
    centers = []
    r_init = 0.06
    row_counts = [5, 5, 5, 5, 6]  # Sums to 26
    y = r_init + 0.05
    for count in row_counts:
        # Center rows horizontally
        x_start = r_init + (1.0 - count * 2 * r_init) / 2.0
        for i in range(count):
            centers.append([x_start + i * 2 * r_init, y])
        y += r_init * np.sqrt(3)
    centers = np.array(centers)
    
    # Break symmetry with small random perturbation to aid optimization
    np.random.seed(42)
    centers += np.random.uniform(-0.005, 0.005, centers.shape)
    centers = np.clip(centers, r_init, 1 - r_init)

    # 2. Setup and run optimization
    initial_params = np.concatenate([centers.flatten(), [r_init]])
    bounds = [(0.0, 1.0)] * (2*n) + [(0.05, 0.5)]
    
    res = minimize(_packing_objective, initial_params, args=(n,), method='L-BFGS-B', 
                   bounds=bounds, options={'maxiter': 10000, 'ftol': 1e-10, 'gtol': 1e-8})
                   
    final_centers = res.x[:2*n].reshape(n, 2)
    
    # 3. Project to guaranteed valid configuration
    # Calculate the tightest constraint from the optimized positions
    diff = final_centers[:, np.newaxis, :] - final_centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Valid radius is limited by closest pair and closest boundary
    valid_r = min_pair_dist / 2.0
    valid_r = min(valid_r, np.min(final_centers[:, 0]))
    valid_r = min(valid_r, 1.0 - np.max(final_centers[:, 0]))
    valid_r = min(valid_r, np.min(final_centers[:, 1]))
    valid_r = min(valid_r, 1.0 - np.max(final_centers[:, 1]))
    
    # Apply tiny safety margin to strictly satisfy the 1e-12 checker tolerance
    final_r = valid_r * 0.99999
    
    # Prepare output
    radii = np.full(n, final_r)
    sum_radii = np.sum(radii)
    
    return final_centers, radii, sum_radii
