# sol_000282 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b0b06613) state=b48344c3 sum of radii=2.390655 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars_flat, n, penalty_weight):
    """
    Computes the objective value: -sum(radii) + penalty_weight * (boundary_violations + overlap_violations)
    """
    centers = vars_flat[:2*n].reshape(n, 2)
    radii = vars_flat[2*n:]
    
    # Base objective: maximize sum of radii
    obj = -np.sum(radii)
    
    # Pairwise overlap penalty
    # Precompute upper triangular indices for efficiency
    ii, jj = np.triu_indices(n, k=1)
    diffs = centers[ii] - centers[jj]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    
    overlaps = radii[ii] + radii[jj] - dists
    pen_overlap = np.sum(np.maximum(0, overlaps)**2)
    
    # Boundary penalty
    # x >= r  => r - x <= 0
    # x <= 1-r => r - (1-x) <= 0
    # y >= r  => r - y <= 0
    # y <= 1-r => r - (1-y) <= 0
    pen_bound = (np.sum(np.maximum(0, radii - centers[:, 0])**2) +
                 np.sum(np.maximum(0, radii - (1 - centers[:, 0]))**2) +
                 np.sum(np.maximum(0, radii - centers[:, 1])**2) +
                 np.sum(np.maximum(0, radii - (1 - centers[:, 1]))**2))
                 
    return obj + penalty_weight * (pen_overlap + pen_bound)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n = 26
    
    # 1. Initialization: Perturbed hexagonal-like grid
    centers = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(6):
            if idx >= n:
                break
            x = 0.05 + j * 0.9 / 5.0 + (i % 2) * 0.09
            y = 0.05 + i * 0.9 / 4.0
            # Add small perturbation to break symmetry
            x += np.random.normal(0, 0.01)
            y += np.random.normal(0, 0.01)
            centers[idx] = [np.clip(x, 0.02, 0.98), np.clip(y, 0.02, 0.98)]
            idx += 1
            
    radii = np.ones(n) * 0.04
    x0 = np.concatenate([centers.ravel(), radii])
    
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # 2. Continuation Optimization
    penalty_weights = [100.0, 1000.0, 5000.0]
    current_vars = x0.copy()
    
    for w in penalty_weights:
        def obj_func(v):
            return compute_objective(v, n, w)
            
        res = minimize(obj_func, current_vars, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 800, 'ftol': 1e-10, 'gtol': 1e-6})
        current_vars = res.x
        
    # Extract results
    final_centers = current_vars[:2*n].reshape(n, 2)
    final_radii = current_vars[2*n:]
    
    # 3. Strict Validity Enforcement (Clipping)
    # This guarantees the packing passes validation without relying on numerical tolerance of the optimizer
    for i in range(n):
        # Boundary constraints
        r_max = min(final_centers[i, 0], 1.0 - final_centers[i, 0],
                    final_centers[i, 1], 1.0 - final_centers[i, 1])
        # Neighbor constraints
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
                r_max = min(r_max, dist / 2.0)
        final_radii[i] = r_max
        
    total_sum = float(np.sum(final_radii))
    return final_centers, final_radii, total_sum
