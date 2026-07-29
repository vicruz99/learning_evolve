# sol_000198 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state cda7e5e4) state=f4ede3f3 sum of radii=2.612787 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(vars[52:])

def constraints(vars):
    """Returns array of inequality constraints (all must be >= 0)"""
    c = vars[:52].reshape(26, 2)
    r = vars[52:]
    
    # 104 boundary constraints
    b_cons = np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])
    
    # 325 pairwise overlap constraints
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    mask = np.triu(np.ones((26, 26), dtype=bool), k=1)
    o_cons = (dist_sq - r_sum_sq)[mask]
    
    return np.concatenate([b_cons, o_cons])

def run_packing() -> tuple:
    np.random.seed(42)
    n = 26
    
    # Phase 1: Hexagonal initialization & Expand/Repel simulation
    centers = np.zeros((n, 2))
    idx = 0
    for row in range(5):
        for col in range(5):
            if idx >= n: break
            centers[idx, 0] = 0.12 + col * 0.18
            centers[idx, 1] = 0.12 + row * 0.18
            if row % 2 == 1:
                centers[idx, 0] += 0.09
            idx += 1
        if idx >= n: break
        
    # Normalize to fit within square
    min_c = centers.min(axis=0)
    max_c = centers.max(axis=0)
    centers = (centers - min_c) / (max_c - min_c) * 0.82 + 0.09
    radii = np.ones(n) * 0.05
    
    # Iterative expansion with overlap resolution
    for step in range(2500):
        # Decreasing growth rate for finer convergence
        growth = 1.0 + 0.0004 * (1.0 - step / 2500.0)
        radii *= growth
        
        # Resolve overlaps via repulsion
        for _ in range(20):
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-14)
            np.fill_diagonal(dists, np.inf)
            
            for i in range(n):
                for j in range(i + 1, n):
                    d = dists[i, j]
                    r_sum = radii[i] + radii[j]
                    if d < r_sum:
                        overlap = r_sum - d
                        vec = centers[i] - centers[j]
                        vec /= d
                        # Push apart symmetrically
                        move = vec * overlap * 0.5
                        centers[i] += move
                        centers[j] -= move
                        
            # Update distances after moves
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-14)
            np.fill_diagonal(dists, np.inf)
            
        # Project to boundaries
        for i in range(n):
            if centers[i, 0] < radii[i]: centers[i, 0] = radii[i]
            if centers[i, 0] > 1.0 - radii[i]: centers[i, 0] = 1.0 - radii[i]
            if centers[i, 1] < radii[i]: centers[i, 1] = radii[i]
            if centers[i, 1] > 1.0 - radii[i]: centers[i, 1] = 1.0 - radii[i]
            
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0) for _ in range(52)] + [(0.0, 1.0) for _ in range(26)]
    cons = {'type': 'ineq', 'fun': constraints}
    
    # Phase 2: SLSQP Polynomial Polish
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 150, 'ftol': 1e-12, 'disp': False})
        if res.success or not np.isnan(res.fun):
            x_opt = res.x
            centers = x_opt[:52].reshape(26, 2)
            radii = x_opt[52:]
    except Exception:
        pass
        
    # Final safety clamp
    radii = np.maximum(radii, 0.0)
    centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
    centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
    
    return centers, radii, float(np.sum(radii))
