# sol_000194 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 320c78c6) state=d01bfd98 sum of radii=2.540000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_function(vars, penalty_weight):
    """
    Objective function to minimize.
    Minimizes -r (maximizing radius) plus penalty for overlaps and boundary violations.
    
    Args:
        vars: Array of shape (2*n + 1), containing centers and radius.
        penalty_weight: Weight for the constraint penalty.
    """
    n = 26
    c = vars[:2*n].reshape((n, 2))
    r = vars[2*n]
    
    # We want to maximize r, so we minimize -r
    val = -r
    p = 0.0
    
    # Pairwise distances
    # Compute distance matrix efficiently
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Consider only upper triangle (i < j) to avoid double counting and self-distance
    mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    pair_dists = dists[mask]
    
    # Overlap penalty: max(0, 2r - dist)^2
    # Violation occurs if 2r > dist
    violations = 2.0 * r - pair_dists
    mask_pos = violations > 0
    p += np.sum(violations[mask_pos]**2)
    
    # Boundary penalties
    # Constraints: r <= x <= 1-r  and  r <= y <= 1-r
    # Violations: r - x > 0 (left), x - (1-r) > 0 (right), etc.
    x = c[:, 0]
    y = c[:, 1]
    
    p += np.sum(np.maximum(0, r - x)**2)
    p += np.sum(np.maximum(0, x - (1.0 - r))**2)
    p += np.sum(np.maximum(0, r - y)**2)
    p += np.sum(np.maximum(0, y - (1.0 - r))**2)
    
    return val + penalty_weight * p

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    
    # 1. Initialization
    # Start with a hexagonal grid pattern which is dense
    centers = np.zeros((n, 2))
    r_init = 0.1
    # Row counts summing to 26: 5, 4, 5, 4, 5, 3
    row_counts = [5, 4, 5, 4, 5, 3]
    
    idx = 0
    y = r_init
    for row, count in enumerate(row_counts):
        # Shift odd rows to create hexagonal packing
        x = r_init + (row % 2) * r_init 
        for col in range(count):
            if idx < n:
                centers[idx] = [x, y]
                x += 2 * r_init
                idx += 1
        y += np.sqrt(3) * r_init
    
    # Normalize and scale to fit in unit square with margin
    min_c = centers.min(axis=0)
    max_c = centers.max(axis=0)
    centers -= min_c
    span = max_c - min_c
    if span.max() > 0:
        centers /= span.max()
    centers *= 0.9 
    centers += 0.05
    
    # 2. Optimization
    # Variables: [c1x, c1y, ..., c26x, c26y, r]
    current_vars = np.concatenate([centers.flatten(), [0.05]])
    
    # Bounds: centers in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * n)
    bounds.append((0, 0.5))
    
    penalty_weight = 100.0
    
    # Iteratively increase penalty weight to enforce constraints strictly
    for _ in range(15):
        res = minimize(objective_function, current_vars, args=(penalty_weight,), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10})
        current_vars = res.x
        penalty_weight *= 5.0

    # Extract results
    final_centers = current_vars[:2*n].reshape((n, 2))
    final_r = current_vars[2*n]
    final_radii = np.full(n, final_r)
    
    # 3. Validation and Fallback
    # Check for validity internally
    valid = True
    for i in range(n):
        r = final_radii[i]
        x, y = final_centers[i]
        # Check bounds
        if x < -1e-12 or x > 1 + 1e-12 or y < -1e-12 or y > 1 + 1e-12:
            valid = False
        # Check circle inside square
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            valid = False
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
            if dist < final_radii[i] + final_radii[j] - 1e-12:
                valid = False
                
    if not valid:
        # Fallback to a known valid configuration
        # 25 circles in 5x5 grid with r=0.1, plus 1 small circle in a gap
        centers_fb = []
        radii_fb = []
        for r in range(5):
            for c in range(5):
                centers_fb.append([0.1 + c*0.2, 0.1 + r*0.2])
                radii_fb.append(0.1)
        # Place 26th circle at (0.2, 0.2) with radius 0.04
        # Distance to nearest grid point (0.1, 0.1) is sqrt(0.02) ~ 0.1414
        # 0.1 + 0.04 < 0.1414, so valid.
        centers_fb.append([0.2, 0.2])
        radii_fb.append(0.04)
        final_centers = np.array(centers_fb)
        final_radii = np.array(radii_fb)

    return final_centers, final_radii, np.sum(final_radii)
