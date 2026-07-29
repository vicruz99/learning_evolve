# sol_000062 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bb78642d) state=641dbca4 sum of radii=2.597361 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def packing_obj(theta, mu, N, upper_tri_idx):
    """Objective function: maximize sum of radii via penalty method."""
    xc = theta[0::3]
    yc = theta[1::3]
    r = theta[2::3]
    
    obj = -np.sum(r)
    pen = 0.0
    
    # Boundary penalties
    pen += np.sum(np.maximum(0, r - xc)**2)
    pen += np.sum(np.maximum(0, r - (1 - xc))**2)
    pen += np.sum(np.maximum(0, r - yc)**2)
    pen += np.sum(np.maximum(0, r - (1 - yc))**2)
    
    # Overlap penalties
    dx = xc[:, None] - xc[None, :]
    dy = yc[:, None] - yc[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    r_sum = r[:, None] + r[None, :]
    v = r_sum[upper_tri_idx] - dist[upper_tri_idx]
    pen += np.sum(np.maximum(0, v)**2)
    
    return obj + mu * pen

def run_packing():
    N = 26
    upper_tri_idx = np.triu_indices(N, k=1)
    
    # 1. Initial hexagonal layout
    centers = []
    s = 0.16
    y = 0.05
    row = 0
    while len(centers) < N:
        x = (s / 2) if row % 2 == 1 else 0.05
        while x <= 0.95 and len(centers) < N:
            centers.append([x, y])
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
        
    centers = np.array(centers[:N])
    # Center in square
    cx, cy = centers.mean(axis=0)
    centers[:, 0] -= cx - 0.5
    centers[:, 1] -= cy - 0.5
    centers = np.clip(centers, 0.05, 0.95)
    
    r_init = np.full(N, 0.08)
    theta0 = np.hstack([centers[:, 0], centers[:, 1], r_init])
    
    bounds = [(0, 1)] * (2 * N) + [(1e-5, 0.5)] * N
    
    theta = theta0.copy()
    mu = 1.0
    
    # 2. Sequential optimization with increasing penalty weight
    for _ in range(25):
        res = minimize(packing_obj, theta, args=(mu, N, upper_tri_idx), 
                       method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1500, 'ftol': 1e-12})
        theta = res.x
        mu *= 2.0
        
    xc = theta[0::3]
    yc = theta[1::3]
    r = theta[2::3]
    
    # 3. Projection to strict feasibility
    # Boundary constraints
    for i in range(N):
        r[i] = min(r[i], xc[i], 1 - xc[i], yc[i], 1 - yc[i])
        
    # Overlap resolution
    for _ in range(50):
        overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(xc[i] - xc[j], yc[i] - yc[j])
                if d < r[i] + r[j]:
                    shrink = (r[i] + r[j] - d) / 2.0
                    r[i] -= shrink
                    r[j] -= shrink
                    overlap = True
        if not overlap:
            break
            
    r = np.maximum(r, 0)
    centers = np.column_stack((xc, yc))
    return centers, r, np.sum(r)
