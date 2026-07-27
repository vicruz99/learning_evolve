# sol_000081 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9821b492) state=62d84410 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss_and_grad(params, n=26):
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Compute pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    dists = np.maximum(dists, 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    # Compute overlaps
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    deltas = rad_sum - dists
    mask = deltas > 0.0
    deltas_active = np.where(mask, deltas, 0.0)
    
    # Overlap penalty
    o_pen = np.sum(deltas_active**2)
    
    # Boundary penalties
    left = radii - centers[:, 0]
    right = radii - (1.0 - centers[:, 0])
    btop = radii - centers[:, 1]
    bott = radii - (1.0 - centers[:, 1])
    
    b_pen = np.sum(np.maximum(left, 0.0)**2 + np.maximum(right, 0.0)**2 + 
                   np.maximum(btop, 0.0)**2 + np.maximum(bott, 0.0)**2)
                   
    lam = 5000.0
    loss = -np.sum(radii) + lam * (b_pen + o_pen)
    
    # Gradients
    grad = np.zeros_like(params)
    
    # d(loss)/d(radii)
    db_dr = 2.0 * (np.maximum(left, 0.0) + np.maximum(right, 0.0) + 
                   np.maximum(btop, 0.0) + np.maximum(bott, 0.0))
    do_dr = 2.0 * np.sum(deltas_active, axis=1)
    grad[2*n:] = -1.0 + lam * (db_dr + do_dr)
    
    # d(loss)/d(x_i)
    db_dx = -2.0 * np.maximum(left, 0.0) + 2.0 * np.maximum(right, 0.0)
    dir_x = diffs[:, :, 0] / dists
    do_dx = 2.0 * np.sum(deltas_active * dir_x, axis=1)
    grad[:n] = lam * (db_dx + do_dx)
    
    # d(loss)/d(y_i)
    db_dy = -2.0 * np.maximum(btop, 0.0) + 2.0 * np.maximum(bott, 0.0)
    dir_y = diffs[:, :, 1] / dists
    do_dy = 2.0 * np.sum(deltas_active * dir_y, axis=1)
    grad[n:2*n] = lam * (db_dy + do_dy)
    
    return loss, grad

def run_packing():
    n = 26
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    starts = []
    
    # 1. Random starts
    for _ in range(10):
        c = np.random.rand(n, 2) * 0.8 + 0.1
        r = np.full(n, 0.05) + np.random.rand(n) * 0.02
        starts.append(np.concatenate([c.ravel(), r]))
        
    # 2. Hexagonal-ish starts
    for _ in range(5):
        c = np.zeros((n, 2))
        y, x, idx = 0.1, 0.1, 0
        for i in range(5):
            for j in range(5):
                if idx >= n: break
                c[idx, 0] = x + j * 0.2 + (i % 2) * 0.1
                c[idx, 1] = y + i * 0.25
                idx += 1
            if idx >= n: break
        while idx < n:
            c[idx] = np.random.rand(2) * 0.8 + 0.1
            idx += 1
        c += np.random.randn(n, 2) * 0.02
        c = np.clip(c, 0.05, 0.95)
        r = np.full(n, 0.09)
        starts.append(np.concatenate([c.ravel(), r]))
        
    # 3. Grid start
    c = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx >= n: break
            c[idx] = [0.1 + j*0.2, 0.1 + i*0.2]
            idx += 1
    c[idx] = [0.6, 0.6]
    c += np.random.randn(n, 2) * 0.01
    r = np.full(n, 0.1)
    starts.append(np.concatenate([c.ravel(), r]))

    for x0 in starts:
        res = minimize(compute_loss_and_grad, x0, method='L-BFGS-B', 
                       bounds=bounds, jac=True, options={'maxiter': 3000, 'ftol': 1e-15})
        
        centers = res.x[:2*n].reshape(n, 2)
        radii = res.x[2*n:]
        
        # Quick validation check
        valid = True
        for k in range(n):
            x, y, r = centers[k, 0], centers[k, 1], radii[k]
            if r < -1e-9 or x - r < -1e-7 or x + r > 1 + 1e-7 or y - r < -1e-7 or y + r > 1 + 1e-7:
                valid = False
                break
        if valid:
            for k1 in range(n):
                for k2 in range(k1+1, n):
                    d = np.hypot(centers[k1,0]-centers[k2,0], centers[k1,1]-centers[k2,1])
                    if d < radii[k1] + radii[k2] - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
                    
        if valid:
            s = np.sum(radii)
            if s > best_sum:
                best_sum = s
                best_centers = centers.copy()
                best_radii = radii.copy()
                
    if best_centers is None:
        best_centers = np.zeros((n, 2))
        best_radii = np.zeros(n)
        best_sum = 0.0
    else:
        # Slight shrinkage to guarantee strict validity under tolerance
        best_radii *= 0.9999999
        
    return best_centers, best_radii, np.sum(best_radii)
