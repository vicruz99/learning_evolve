# sol_000108 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a3c1a30f) state=80d1dddc sum of radii=2.603126 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n, weight):
    """
    Computes the objective function: -sum_radii + penalty * violations
    """
    centers = vars[:2*n].reshape((n, 2))
    radii = vars[2*n:]
    
    sum_r = np.sum(radii)
    penalty = 0.0
    
    # Boundary penalties (vectorized)
    r = radii
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Check boundaries: r <= x, r <= 1-x, r <= y, r <= 1-y
    penalty += np.sum(np.maximum(0, r - x)**2)
    penalty += np.sum(np.maximum(0, r - (1 - x))**2)
    penalty += np.sum(np.maximum(0, r - y)**2)
    penalty += np.sum(np.maximum(0, r - (1 - y))**2)
    
    # Distance penalties
    # Compute pairwise squared distances using broadcasting
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    
    # Min squared distance required for non-overlap
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = r_sum**2
    
    # Violation: min_dist_sq - dist_sq > 0 means overlap
    violation = np.maximum(0, min_dist_sq - dist_sq)
    penalty += np.sum(violation**2)
    
    return -sum_r + weight * penalty

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    # Initialize centers in a hexagonal-like pattern
    # Rows: 5, 6, 5, 6, 4 circles
    row_counts = [5, 6, 5, 6, 4]
    centers = []
    sx, sy = 0.18, 0.16
    y_pos = 0.1
    
    for idx, count in enumerate(row_counts):
        x_start = (1.0 - (count - 1) * sx) / 2.0
        for k in range(count):
            centers.append([x_start + k * sx, y_pos])
        y_pos += sy
        
    centers = np.array(centers)
    centers = np.clip(centers, 0.0, 1.0)
    radii = np.full(n, 0.08)
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Continuation method with increasing penalty weight
    current_x = x0
    weights = [10.0, 100.0, 1000.0, 10000.0]
    
    for w in weights:
        res = minimize(compute_objective, current_x, args=(n, w), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-12})
        current_x = res.x
        
    final_centers = current_x[:2*n].reshape((n, 2))
    final_radii = current_x[2*n:]
    
    # Post-processing to guarantee validity within tolerance
    # 1. Boundary constraints
    for i in range(n):
        x, y = final_centers[i]
        r = final_radii[i]
        r = min(r, x, 1 - x, y, 1 - y)
        final_radii[i] = r
        
    # 2. Pairwise distance constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = final_centers[i, 0] - final_centers[j, 0]
            dy = final_centers[i, 1] - final_centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_sum = final_radii[i] + final_radii[j]
            if dist < min_sum:
                diff = min_sum - dist
                final_radii[i] -= diff / 2
                final_radii[j] -= diff / 2
                
    final_radii = np.maximum(final_radii, 0.0)
    sum_r = np.sum(final_radii)
    
    return final_centers, final_radii, sum_r
