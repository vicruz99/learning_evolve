# sol_000074 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfef56bb) state=0aa3c968 sum of radii=2.577772 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(centers, radii, alpha):
    """Compute boundary and overlap penalties for the packing configuration."""
    n = len(radii)
    pen = 0.0
    
    # Boundary penalties: circles must stay within [0, 1]
    pen += alpha * np.sum(np.maximum(0, radii - centers[:, 0])**2)
    pen += alpha * np.sum(np.maximum(0, radii - (1 - centers[:, 0]))**2)
    pen += alpha * np.sum(np.maximum(0, radii - centers[:, 1])**2)
    pen += alpha * np.sum(np.maximum(0, radii - (1 - centers[:, 1]))**2)
    
    # Overlap penalties: circles must not intersect
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    overlap = np.maximum(0, r_sum - dists)
    # Only count each pair once (upper triangle)
    overlap = np.triu(overlap, k=1)
    pen += alpha * np.sum(overlap**2)
    
    return pen

def objective_function(vars, n, alpha):
    """Objective to minimize: -sum(radii) + penalty."""
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    return -np.sum(radii) + compute_penalty(centers, radii, alpha)

def run_packing():
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    n = 26
    np.random.seed(42)  # Deterministic initialization
    
    # 1. Hexagonal lattice initialization for dense starting configuration
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.04)
    
    idx = 0
    row, col = 0, 0
    spacing = 0.22
    while idx < n:
        x = col * spacing + (0.5 * spacing if row % 2 == 1 else 0) + spacing/2
        y = row * spacing * np.sqrt(3)/2 + spacing/2
        centers[idx] = [x, y]
        idx += 1
        col += 1
        if col > 5:
            col = 0
            row += 1
            
    # Scale to fit comfortably inside the square and add small jitter
    cmin, cmax = centers.min(axis=0), centers.max(axis=0)
    centers = (centers - cmin) / (cmax - cmin) * 0.7 + 0.15
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    
    # Flatten parameters for optimizer: [x1, y1, ..., x26, y26, r1, ..., r26]
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)]*(2*n) + [(0.0, 0.5)]*n
    
    # 2. Homotopy continuation: gradually increase penalty weight
    alphas = [100.0, 1000.0, 5000.0, 20000.0]
    curr_vars = x0.copy()
    
    for alpha in alphas:
        res = minimize(objective_function, curr_vars, args=(n, alpha), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-8})
        curr_vars = res.x
        
    final_centers = curr_vars[:2*n].reshape(n, 2)
    final_radii = curr_vars[2*n:]
    
    # 3. Deterministic constraint satisfaction post-processing
    for _ in range(30):
        changed = False
        # Enforce boundary constraints
        for i in range(n):
            x, y = final_centers[i]
            r = final_radii[i]
            for b in [x, 1-x, y, 1-y]:
                if r > b - 1e-9:
                    final_radii[i] = max(0, b - 1e-9)
                    changed = True
                    
        # Enforce non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                dist = np.hypot(final_centers[i,0]-final_centers[j,0], 
                                final_centers[i,1]-final_centers[j,1])
                if dist < final_radii[i] + final_radii[j] - 1e-9:
                    excess = final_radii[i] + final_radii[j] - dist
                    final_radii[i] -= excess/2
                    final_radii[j] -= excess/2
                    changed = True
                    
        if not changed:
            break
            
    # Ensure strictly positive radii
    final_radii = np.maximum(final_radii, 1e-9)
    
    return final_centers, final_radii, np.sum(final_radii)
