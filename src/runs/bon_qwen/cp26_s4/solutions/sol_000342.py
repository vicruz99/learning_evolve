# sol_000342 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2bd19375) state=757d550d sum of radii=1.560000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_loss(v, N):
    """Computes the violation penalty for boundary and pairwise overlap constraints."""
    C = v[:2*N].reshape(N, 2)
    R = v[2*N:]
    
    # Boundary penalties: r <= x <= 1-r, r <= y <= 1-r
    pen = np.sum(np.maximum(0, R - C[:,0])**2 + 
                 np.maximum(0, C[:,0] + R - 1)**2 + 
                 np.maximum(0, R - C[:,1])**2 + 
                 np.maximum(0, C[:,1] + R - 1)**2)
    
    # Pairwise overlap penalties: dist >= r_i + r_j
    # Vectorized distance calculation
    dists = np.sqrt(np.sum((C[:, None, :] - C[None, :, :])**2, axis=2))
    r_sums = R[:, None] + R[None, :]
    overlaps = np.maximum(0, r_sums - dists)
    
    # Sum only upper triangle to avoid double counting
    triu_mask = np.triu(np.ones((N, N)), k=1)
    pen += np.sum(overlaps**2 * triu_mask)
    
    return pen

def objective(v, N, lam):
    """Objective to minimize: negative sum of radii + penalty * constraint violations."""
    return -np.sum(v[2*N:]) + lam * compute_loss(v, N)

def get_initial_guess(N, seed):
    """Generates an initial configuration based on a hexagonal pattern."""
    rng = np.random.default_rng(seed)
    centers = []
    r_base = 0.09
    
    # Hexagonal arrangement: rows with 6, 5, 6, 5, 4 circles sum to 26
    pattern = [6, 5, 6, 5, 4]
    idx = 0
    for row, count in enumerate(pattern):
        y = r_base + row * np.sqrt(3) * r_base
        for col in range(count):
            x = r_base + (col + (row % 2) * 0.5) * 2 * r_base
            centers.append([x, y])
            idx += 1
            if idx >= N:
                break
        if idx >= N:
            break
            
    # Fallback fill if pattern is short (shouldn't happen with N=26)
    while len(centers) < N:
        centers.append([rng.random(), rng.random()])
        
    centers = np.array(centers[:N])
    # Normalize to fit safely inside [0,1] with padding
    centers = centers.min(axis=0) + (centers - centers.min(axis=0)) * 0.8
    
    radii = np.full(N, 0.06)
    return np.concatenate([centers.flatten(), radii])

def run_packing():
    N = 26
    best_v = None
    best_loss = np.inf
    
    # Bounds ensure centers stay away from exact edges and radii stay positive
    bounds = [(0.005, 0.995) for _ in range(2*N)] + [(0.005, 0.5) for _ in range(N)]
    
    # Multiple restarts to escape local minima
    for seed in range(5):
        v0 = get_initial_guess(N, seed)
        
        # Phase 1: Moderate penalty to grow radii and arrange circles
        res = minimize(objective, v0, args=(N, 1500.0), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-10})
        loss = compute_loss(res.x, N)
        
        if loss < best_loss:
            best_loss = loss
            best_v = res.x.copy()
            
        # Phase 2: High penalty to strictly resolve any remaining overlaps
        if loss > 1e-4:
            res2 = minimize(objective, res.x, args=(N, 8000.0), method='L-BFGS-B',
                            bounds=bounds, options={'maxiter': 500, 'ftol': 1e-10})
            loss2 = compute_loss(res2.x, N)
            if loss2 < best_loss:
                best_loss = loss2
                best_v = res2.x.copy()
                
    opt_centers = best_v[:2*N].reshape(N, 2)
    opt_radii = best_v[2*N:]
    
    # Ensure radii are non-negative
    opt_radii = np.maximum(opt_radii, 0.0)
    
    return opt_centers, opt_radii, np.sum(opt_radii)
