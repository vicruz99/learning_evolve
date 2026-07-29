# sol_000071 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a76a0e24) state=90cdc2a1 sum of radii=2.262834 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(z, N):
    """
    Objective function: maximize sum of radii (minimize negative sum)
    with penalty terms for boundary and overlap constraints.
    """
    centers = z[:2 * N].reshape(N, 2)
    radii = z[2 * N:]
    
    loss = 0.0
    
    # Boundary penalties
    for i in range(N):
        cx, cy = centers[i]
        r = radii[i]
        
        # Left & Right
        if cx < r:
            loss += 1000.0 * (r - cx) ** 2
        if cx > 1.0 - r:
            loss += 1000.0 * (cx - (1.0 - r)) ** 2
            
        # Top & Bottom
        if cy < r:
            loss += 1000.0 * (r - cy) ** 2
        if cy > 1.0 - r:
            loss += 1000.0 * (cy - (1.0 - r)) ** 2
            
    # Overlap penalties
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            req = radii[i] + radii[j]
            if dist < req:
                loss += 2000.0 * (req - dist) ** 2
                
    return -np.sum(radii) + loss

def run_packing():
    N = 26
    
    # 1. Initialize centers in a hexagonal lattice pattern
    pts = []
    for j in range(6):
        for i in range(6):
            x = i * 2.0 + (j % 2)
            y = j * np.sqrt(3)
            pts.append([x, y])
            
    pts = np.array(pts[:N])
    
    # Normalize to fit roughly inside [0.1, 0.9]
    pts -= pts.min(axis=0)
    pts /= pts.max(axis=0)
    pts *= 0.75
    pts += 0.125
    
    # Initial radii
    radii = np.full(N, 0.07)
    
    # Flatten to optimization variable vector
    x0 = np.concatenate([pts.flatten(), radii])
    
    # Bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    # 2. Optimize
    res = minimize(
        compute_loss, 
        x0, 
        args=(N,), 
        bounds=bounds, 
        method='L-BFGS-B', 
        options={'maxiter': 3000, 'ftol': 1e-10}
    )
    
    final_centers = res.x[:2 * N].reshape(N, 2)
    final_radii = res.x[2 * N:]
    
    # 3. Post-optimization safety adjustments to guarantee validity
    for i in range(N):
        cx, cy = final_centers[i]
        ri = final_radii[i]
        # Enforce boundary constraints strictly
        final_radii[i] = min(ri, cx, 1.0 - cx, cy, 1.0 - cy, 0.5)
        
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt(np.sum((final_centers[i] - final_centers[j]) ** 2))
            req = final_radii[i] + final_radii[j]
            if dist < req:
                # Shrink overlapping circles slightly to resolve violation
                delta = (req - dist) / 2.0 + 1e-7
                final_radii[i] -= delta
                final_radii[j] -= delta
                
    return final_centers, final_radii, np.sum(final_radii)
