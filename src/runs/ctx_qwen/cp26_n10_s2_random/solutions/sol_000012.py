# sol_000012 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8a979775) state=46f5cf56 sum of radii=2.127796 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(vars, n, lam):
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    pen = 0.0
    
    # Boundary constraint penalties
    # Left: x >= r
    pen += np.sum(np.maximum(0, r - c[:, 0])**2)
    # Right: x + r <= 1
    pen += np.sum(np.maximum(0, c[:, 0] + r - 1)**2)
    # Bottom: y >= r
    pen += np.sum(np.maximum(0, r - c[:, 1])**2)
    # Top: y + r <= 1
    pen += np.sum(np.maximum(0, c[:, 1] + r - 1)**2)
    
    # Overlap constraint penalties
    c1 = c[:, np.newaxis, :]
    c2 = c[np.newaxis, :, :]
    dist = np.sqrt(np.sum((c1 - c2)**2, axis=2) + 1e-16)
    r1 = r[:, np.newaxis]
    r2 = r[np.newaxis, :]
    overlap = r1 + r2 - dist
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    pen += np.sum(np.maximum(0, overlap[mask])**2)
    
    return -np.sum(r) + lam * pen

def run_packing():
    n = 26
    r_init = 0.08
    centers = []
    radii = [r_init] * n
    
    # Hexagonal lattice initialization: 6, 5, 6, 5, 4 circles per row
    row_counts = [6, 5, 6, 5, 4]
    for row in range(5):
        y = r_init + row * np.sqrt(3) * r_init
        count = row_counts[row]
        x_start = r_init if row % 2 == 0 else 2 * r_init
        for col in range(count):
            x = x_start + col * 2 * r_init
            centers.append([x, y])
            
    centers = np.array(centers)
    # Break symmetry with small random noise to escape degenerate local minima
    np.random.seed(42)
    centers += np.random.normal(0, 0.005, centers.shape)
    
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0, 1)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # Optimization with high penalty to enforce constraints
    lam = 20000.0
    res = minimize(compute_penalty, x0, args=(n, lam), method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-12})
                   
    opt_centers = res.x[:2*n].reshape(n, 2)
    opt_radii = res.x[2*n:]
    
    # Post-processing to strictly enforce constraints and handle numerical drift
    for _ in range(10):
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((opt_centers[i] - opt_centers[j])**2))
                req = opt_radii[i] + opt_radii[j]
                if d < req - 1e-12:
                    shrink = (req - d) / 2 + 1e-9
                    opt_radii[i] -= shrink
                    opt_radii[j] -= shrink
            for k in range(2):
                if opt_centers[i][k] - opt_radii[i] < -1e-12:
                    opt_radii[i] = opt_centers[i][k] - 1e-9
                if opt_centers[i][k] + opt_radii[i] > 1 + 1e-12:
                    opt_radii[i] = 1 - opt_centers[i][k] - 1e-9
        opt_radii = np.maximum(opt_radii, 1e-9)
        
    total_r = np.sum(opt_radii)
    return opt_centers, opt_radii, total_r
