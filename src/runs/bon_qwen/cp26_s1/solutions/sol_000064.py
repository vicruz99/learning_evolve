# sol_000064 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cae61cda) state=969df99b sum of radii=2.583028 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < -1e-12:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    
    # --- 1. Initial Configuration ---
    # Hexagonal packing rows: 5, 6, 5, 6, 4 sums to 26
    row_counts = [5, 6, 5, 6, 4]
    centers = []
    s = 0.21 # Estimated spacing
    y = 0.1
    
    for i, cnt in enumerate(row_counts):
        x_start = 0.5 - (cnt - 1) * s / 2
        if i % 2 == 1:
            x_start += s / 2.0
        for k in range(cnt):
            centers.append([x_start + k * s, y])
        y += s * np.sqrt(3) / 2.0
        
    centers = np.array(centers)
    
    # Normalize to fit well in [0,1]
    mins = np.min(centers, axis=0)
    maxs = np.max(centers, axis=0)
    span = maxs - mins
    span[span < 1e-9] = 1e-9
    
    # Map to [0.05, 0.95] to leave room for radii optimization
    centers = (centers - mins) / span
    centers = centers * 0.9 + 0.05
    
    # Initial radii
    r_init = 0.08 * np.ones(N)
    
    # Flatten to vector for optimizer: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(N * 3)
    for i in range(N):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = r_init[i]
        
    bounds = [(0, 1)] * (3 * N)
    
    # --- 2. Constraints ---
    # Boundary: x >= r, 1-x >= r, y >= r, 1-y >= r
    def con_bound(v):
        res = np.empty(N * 4)
        for i in range(N):
            xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
            res[4*i] = xi - ri
            res[4*i+1] = 1 - xi - ri
            res[4*i+2] = yi - ri
            res[4*i+3] = 1 - yi - ri
        return res
        
    # Overlap: dist >= r_i + r_j
    def con_overlap(v):
        M = N * (N - 1) // 2
        res = np.empty(M)
        idx = 0
        for i in range(N):
            xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
            for j in range(i+1, N):
                xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx*dx + dy*dy)
                res[idx] = dist - (ri + rj)
                idx += 1
        return res
        
    cons = [
        {'type': 'ineq', 'fun': con_bound},
        {'type': 'ineq', 'fun': con_overlap}
    ]
    
    # --- 3. Objective ---
    def obj(v):
        return -np.sum(v[2::3])
        
    # --- 4. Optimization ---
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 500, 'ftol': 1e-10, 'disp': False})
    
    # --- 5. Post-process ---
    best_centers = np.zeros((N, 2))
    best_radii = np.zeros(N)
    for i in range(N):
        best_centers[i] = [res.x[3*i], res.x[3*i+1]]
        best_radii[i] = max(0, res.x[3*i+2]) # Ensure non-negative
        
    # Fallback check
    if not validate_packing(best_centers, best_radii):
        best_centers = centers
        best_radii = r_init
        
    return best_centers, best_radii, float(np.sum(best_radii))
