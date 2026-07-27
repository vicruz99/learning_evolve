# sol_000009 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6882cd8b) state=b63827af sum of radii=2.452886 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(vars, penalty, n):
    """Vectorized loss function for circle packing optimization."""
    C = vars[:2 * n].reshape(n, 2)
    R = vars[2 * n:]
    
    # Objective: maximize sum of radii
    loss = -np.sum(R)
    
    # Boundary penalties: ensure circles stay inside [0,1]x[0,1]
    # x - r >= 0  =>  r - x <= 0
    loss += penalty * np.sum(np.maximum(0, R - C[:, 0]) ** 2)
    # x + r <= 1  =>  r - (1-x) <= 0
    loss += penalty * np.sum(np.maximum(0, R - (1 - C[:, 0])) ** 2)
    # y - r >= 0
    loss += penalty * np.sum(np.maximum(0, R - C[:, 1]) ** 2)
    # y + r <= 1
    loss += penalty * np.sum(np.maximum(0, R - (1 - C[:, 1])) ** 2)
    # r >= 0
    loss += penalty * np.sum(np.maximum(0, -R) ** 2)
    
    # Overlap penalties: ensure dist(i,j) >= r_i + r_j
    diff = C[:, None, :] - C[None, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    rad_sums = R[:, None] + R[None, :]
    overlaps = np.maximum(0, rad_sums - dists)
    
    # Only count each pair once
    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    loss += penalty * np.sum(overlaps[triu_mask] ** 2)
    
    return loss

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize centers in a hexagonal-like pattern
    row_counts = [5, 6, 5, 6, 4]
    centers = []
    y = 0.15
    dy = 0.17
    for cnt in row_counts:
        x_vals = np.linspace(0.2, 0.8, cnt)
        for x in x_vals:
            centers.append([x, y])
        y += dy
    centers = np.array(centers)
    
    # Add small perturbation to break symmetry and aid optimization
    np.random.seed(42)
    centers += np.random.uniform(-0.005, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    radii = np.full(n, 0.08)
    x0 = np.concatenate([centers.ravel(), radii])
    
    # Bounds for variables: x in [0,1], y in [0,1], r in [0, 0.25]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.25)] * n
    
    # 2. Multi-stage penalty optimization
    current_x = x0.copy()
    # Gradually increase penalty to first arrange circles, then enforce strict constraints
    for p in [500, 2000, 8000]:
        res = minimize(compute_loss, current_x, args=(p, n), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 1500, 'ftol': 1e-12})
        current_x = res.x
        
    centers_opt = current_x[:2 * n].reshape((n, 2))
    radii_opt = current_x[2 * n:]
    
    # 3. Enforce boundary constraints exactly
    for i in range(n):
        x, y = centers_opt[i]
        r = radii_opt[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y, 0.25)
        radii_opt[i] = min(r, max_r)
        
    # 4. Resolve any residual overlaps via uniform scaling
    scale = 1.0
    for _ in range(100):
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.hypot(centers_opt[i, 0] - centers_opt[j, 0], 
                                centers_opt[i, 1] - centers_opt[j, 1])
                if dist < radii_opt[i] * scale + radii_opt[j] * scale - 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        scale *= 0.95
        
    radii_opt *= scale
    
    # Ensure no negative radii due to numerical noise
    radii_opt = np.maximum(radii_opt, 1e-9)
    
    return centers_opt, radii_opt, float(np.sum(radii_opt))
