# sol_000294 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d1ce3e9) state=27ea876e sum of radii=2.604141 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def generate_initial_guess(n):
    """Generate a feasible hexagonal-like initial configuration."""
    # Pattern of circles per row for n=26 optimized for hexagonal packing density
    rows = [6, 5, 6, 5, 4]
    centers = []
    radii = []
    
    r_init = 0.085
    y = r_init
    
    for nc in rows:
        if nc > 1:
            spacing = (1.0 - 2.0 * r_init) / (nc - 1)
        else:
            spacing = 0.0
            
        x_start = r_init
        for _ in range(nc):
            centers.append([x_start, y])
            radii.append(r_init)
            x_start += spacing
            
        y += spacing * np.sqrt(3.0) / 2.0
        
    return np.array(centers), np.array(radii)

def compute_loss(vars, n, lam):
    """Compute objective + penalty for boundary and overlap constraints."""
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    obj = -np.sum(r)
    
    # Boundary penalties: circles must be inside [0,1]x[0,1]
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    margin_x1 = c[:, 0] - r
    margin_x2 = 1.0 - c[:, 0] - r
    margin_y1 = c[:, 1] - r
    margin_y2 = 1.0 - c[:, 1] - r
    
    obj += lam * np.sum(np.maximum(0.0, -margin_x1)**2)
    obj += lam * np.sum(np.maximum(0.0, -margin_x2)**2)
    obj += lam * np.sum(np.maximum(0.0, -margin_y1)**2)
    obj += lam * np.sum(np.maximum(0.0, -margin_y2)**2)
    
    # Overlap penalties: dist_ij >= r_i + r_j
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    overlap_margin = dists - r_sum
    # Only consider upper triangular part (i < j) to avoid double counting and self-distance
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    obj += lam * np.sum(np.maximum(0.0, -overlap_margin[mask])**2)
    
    return obj

def run_packing():
    n = 26
    centers_init, radii_init = generate_initial_guess(n)
    x0 = np.concatenate([centers_init.flatten(), radii_init])
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    lam = 100.0
    current_vars = x0.copy()
    
    # Iterative penalty method to drive violations to zero
    # Increasing lambda forces the solution to satisfy constraints more strictly
    for _ in range(12):
        res = minimize(compute_loss, current_vars, args=(n, lam), method='L-BFGS-B', 
                       bounds=bounds, options={'ftol': 1e-15, 'gtol': 1e-15, 'maxiter': 500})
        current_vars = res.x
        lam *= 5.0
        
    centers = current_vars[:2*n].reshape(n, 2)
    radii = current_vars[2*n:]
    
    # Final safety clamp to ensure strict validity within float tolerance
    eps = 1e-9
    centers = np.clip(centers, eps, 1.0 - eps)
    radii = np.maximum(radii, eps)
    
    # Adjust radii slightly if any boundary/overlap is violated by tiny numerical drift
    c = centers
    r = radii
    
    # Check boundaries
    min_margin = np.min([np.min(c[:, 0] - r), np.min(1.0 - c[:, 0] - r),
                         np.min(c[:, 1] - r), np.min(1.0 - c[:, 1] - r)])
                     
    # Check overlaps
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    overlap_margin = dists - r_sum
    np.fill_diagonal(overlap_margin, 1.0) # Ignore self-distances
    min_margin = min(min_margin, np.min(overlap_margin))
    
    if min_margin < 0:
        # Scale down radii slightly to guarantee validity against the validator's 1e-12 tolerance
        # We shrink just enough to clear the violation plus a safety buffer
        buffer = 1e-8
        scale = (min_margin + buffer) / (2.0 * np.max(r)) if np.max(r) > 0 else 1.0
        scale = max(scale, 0.99999) # Prevent over-shrinking
        radii *= scale
        
    return centers, radii, np.sum(radii)
