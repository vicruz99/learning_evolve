# sol_000290 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a4dfceb8) state=cda9a1fe sum of radii=2.075705 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def objective(vars, n, lam):
    centers = vars[:2*n].reshape((n, 2))
    radii = vars[2*n:]
    
    # Boundary penalties: circles must stay within [0,1]^2
    pen = np.maximum(0, radii - centers[:, 0])**2
    pen += np.maximum(0, centers[:, 0] + radii - 1)**2
    pen += np.maximum(0, radii - centers[:, 1])**2
    pen += np.maximum(0, centers[:, 1] + radii - 1)**2
    
    # Pairwise overlap penalties
    c_diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(c_diff**2, axis=2))
    r_sum = radii[:, None] + radii[None, :]
    
    triu_idx = np.triu_indices(n, k=1)
    overlap = np.maximum(0, r_sum[triu_idx] - dists[triu_idx])**2
    
    total_pen = np.sum(pen) + np.sum(overlap)
    return -np.sum(radii) + lam * total_pen

def run_packing():
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -1.0
    lam = 5000.0
    
    for seed in range(20):
        np.random.seed(seed)
        
        # Hexagonal lattice initialization
        centers_init = []
        r_init = 0.075
        dy = r_init * np.sqrt(3)
        dx = r_init * 2
        
        y = r_init
        row = 0
        while y < 1 - r_init:
            x = r_init
            if row % 2 == 1:
                x += dx / 2
            while x < 1 - r_init:
                centers_init.append([x, y])
                x += dx
            y += dy
            row += 1
            
        centers_init = np.array(centers_init[:n])
        if len(centers_init) < n:
            extra = n - len(centers_init)
            centers_init = np.vstack([centers_init, np.random.rand(extra, 2) * 0.5 + 0.25])
            
        centers_init += np.random.normal(0, 0.005, size=centers_init.shape)
        radii_init = np.full(n, 0.05)
        
        x0 = np.concatenate([centers_init.flatten(), radii_init])
        bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
        
        res = opt.minimize(objective, x0, args=(n, lam), method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 5000, 'ftol': 1e-12})
                           
        c_res = res.x[:2*n].reshape((n, 2))
        r_res = res.x[2*n:]
        
        curr_sum = np.sum(r_res)
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = c_res.copy()
            best_radii = r_res.copy()
            
    # Strict safety clamping
    best_centers = np.clip(best_centers, 0.0, 1.0)
    best_radii = np.clip(best_radii, 0.0, 0.5)
    
    for i in range(n):
        x, y = best_centers[i]
        best_radii[i] = min(best_radii[i], x, 1-x, y, 1-y)
        
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            if d < best_radii[i] + best_radii[j]:
                overlap = best_radii[i] + best_radii[j] - d
                best_radii[i] -= overlap / 2
                best_radii[j] -= overlap / 2
                best_radii[i] = max(0, best_radii[i])
                best_radii[j] = max(0, best_radii[j])
                
    return best_centers, best_radii, np.sum(best_radii)
