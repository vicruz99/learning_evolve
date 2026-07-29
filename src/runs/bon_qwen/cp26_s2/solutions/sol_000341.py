# sol_000341 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d755ba05) state=c876ab59 sum of radii=2.039397 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(Z, n, pw):
    centers = Z[:2*n].reshape(n, 2)
    radii = Z[2*n:]
    
    # Objective: maximize sum of radii -> minimize negative sum
    loss = -np.sum(radii)
    
    # Boundary penalty: circles must be inside [0,1]^2
    min_edge = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                          np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    b_viol = np.maximum(0.0, radii - min_edge)
    loss += pw * np.sum(b_viol**2)
    
    # Overlap penalty: distance between centers must be >= sum of radii
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    rad_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    triu_idx = np.triu_indices(n, k=1)
    ov_viol = np.maximum(0.0, rad_sums[triu_idx] - dists[triu_idx])
    loss += pw * np.sum(ov_viol**2)
    
    return loss

def run_packing():
    n = 26
    
    # Initialize centers on a shifted hexagonal grid for high density
    xs, ys = [], []
    ry = 0.15
    dy = 0.18
    for r in range(6):
        if r % 2 == 0:
            row_xs = [0.15 + c * 0.2 for c in range(5)]
        else:
            row_xs = [0.25 + c * 0.2 for c in range(4)]
        for x in row_xs:
            xs.append(x)
            ys.append(ry)
        ry += dy
        
    # Ensure exactly n circles
    xs = xs[:n]
    ys = ys[:n]
    
    # Initial radii
    r0 = np.full(n, 0.05)
    Z0 = np.concatenate([xs, ys, r0])
    
    # Bounds for optimization: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    # Progressive penalty optimization
    Z = Z0.copy()
    for pw in [500, 2000, 8000, 20000]:
        res = minimize(compute_loss, Z, args=(n, pw), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 400, 'ftol': 1e-10})
        Z = res.x
        
    centers = Z[:2*n].reshape(n, 2)
    radii = Z[2*n:]
    
    # Post-processing: compute exact max scaling factor to ensure validity
    s = 1.0
    
    # Boundary constraints
    min_edge = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                          np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    valid_r = radii > 1e-9
    if np.any(valid_r):
        s = min(s, np.min(min_edge[valid_r] / radii[valid_r]))
        
    # Overlap constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    rad_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    triu_idx = np.triu_indices(n, k=1)
    ratios = dists[triu_idx] / rad_sums[triu_idx]
    s = min(s, np.min(ratios))
    
    # Apply scaling with a tiny safety margin to satisfy validator's 1e-12 tolerance
    safety = 0.999999
    radii = radii * s * safety
    sum_r = float(np.sum(radii))
    
    return centers, radii, sum_r
