# sol_000129 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 873af2c6) state=8cf7afc5 sum of radii=1.469905 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(v):
    n = 26
    centers = v[:2*n].reshape(n, 2)
    radii = v[2*n:]
    
    # Objective: maximize sum of radii
    loss = -np.sum(radii)
    
    # High penalty weight to enforce constraints
    W = 50000.0
    pen = 0.0
    
    # Boundary constraints
    for i in range(n):
        x, y, r = centers[i,0], centers[i,1], radii[i]
        if r > x: pen += (r - x)**2
        if x + r > 1: pen += (x + r - 1)**2
        if r > y: pen += (r - y)**2
        if y + r > 1: pen += (y + r - 1)**2
        
    # Pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i,0] - centers[j,0]
            dy = centers[i,1] - centers[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_d = radii[i] + radii[j]
            if min_d > dist:
                pen += (min_d - dist)**2
                
    return loss + W * pen

def run_packing():
    n = 26
    
    # 1. Initialize centers in a hexagonal lattice pattern
    pts = []
    dx = 0.12
    dy = dx * np.sqrt(3) / 2
    r_idx = 0
    c_idx = 0
    while len(pts) < n:
        x = c_idx * dx + (r_idx % 2) * dx / 2
        y = r_idx * dy
        pts.append([x, y])
        c_idx += 1
        if c_idx > 7:
            c_idx = 0
            r_idx += 1
            
    centers = np.array(pts[:n])
    
    # Center and scale to fit within [0,1]
    cx, cy = centers.min(axis=0), centers.max(axis=0)
    span = cy - cx
    scale = 0.85 / max(span[0], span[1])
    centers = (centers - cx) * scale + (1 - scale) / 2
    
    # Add deterministic small perturbation to break symmetry
    centers += np.array([[i*0.001, i*0.0013] for i in range(n)])
    
    # Initial radii
    radii = np.full(n, 0.09)
    
    # Flatten for optimizer
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
    
    # 2. Run optimization
    res = minimize(compute_loss, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 5000})
    
    final_centers = res.x[:2*n].reshape(n, 2)
    final_radii = res.x[2*n:]
    
    # 3. Enforce strict validity by shrinking radii if any violation exists
    min_viol = 1e-12
    
    # Check boundaries
    for i in range(n):
        x, y, r = final_centers[i,0], final_centers[i,1], final_radii[i]
        min_viol = max(min_viol, r - x, x + r - 1, r - y, y + r - 1)
        
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dx = final_centers[i,0] - final_centers[j,0]
            dy = final_centers[i,1] - final_centers[j,1]
            dist = np.sqrt(dx*dx + dy*dy)
            min_viol = max(min_viol, final_radii[i] + final_radii[j] - dist)
            
    # Apply shrinkage to guarantee constraints are met
    shrink_amount = max(0, min_viol + 1e-9)
    final_radii -= shrink_amount
    final_radii = np.maximum(final_radii, 0)
    
    return final_centers, final_radii, np.sum(final_radii)
