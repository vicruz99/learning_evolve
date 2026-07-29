# sol_000062 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000035 (state cfcb3616) state=96e8d954 sum of radii=2.624822 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
i_idx, j_idx = np.triu_indices(N, k=1)

def objective(vars):
    """Objective: maximize sum of radii <=> minimize negative sum."""
    return -np.sum(vars[2*N:])

def constraints(vars):
    """
    Inequality constraints: boundary containment and pairwise non-overlap.
    Returns a 1D array where each element must be >= 0.
    """
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    c = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c.append(centers[:, 0] - radii)
    c.append(1.0 - centers[:, 0] - radii)
    c.append(centers[:, 1] - radii)
    c.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = centers[:, 0, np.newaxis] - centers[:, 0]
    dy = centers[:, 1, np.newaxis] - centers[:, 1]
    dist_sq = dx**2 + dy**2
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    c.append(dist_sq[i_idx, j_idx] - r_sum[i_idx, j_idx]**2)
    
    return np.concatenate(c)

def compute_safe_radii(centers):
    """Compute strictly feasible initial radii based on distances to walls and neighbors."""
    dists_wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                            np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_c = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists_c, np.inf)
    dists_min = np.min(dists_c, axis=1)
    
    # Scale down to ensure strict feasibility for the optimizer start
    return np.minimum(dists_wall, 0.5 * dists_min) * 0.85

def run_packing():
    np.random.seed(42)
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_vars = None
    best_sum = -np.inf
    
    inits = []
    
    # 1. Hexagonal lattice variations (rows: 6,5,6,5,4)
    for _ in range(6):
        r_est = 0.08 + 0.02 * np.random.rand()
        rows = [6, 5, 6, 5, 4]
        pts = []
        y = r_est
        for k, cnt in enumerate(rows):
            x_start = r_est if k % 2 == 0 else 2 * r_est
            for m in range(cnt):
                pts.append([x_start + m * 2 * r_est, y])
            y += np.sqrt(3.0) * r_est
        pts = np.array(pts[:N]) + np.random.randn(N, 2) * 0.02
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(np.concatenate([pts.flatten(), compute_safe_radii(pts)]))
        
    # 2. Grid variations (5x5 + center)
    for _ in range(6):
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([0.1 + i * 0.2, 0.1 + j * 0.2])
        pts.append([0.5, 0.5])
        pts = np.array(pts[:N]) + np.random.randn(N, 2) * 0.02
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(np.concatenate([pts.flatten(), compute_safe_radii(pts)]))
        
    # 3. Random uniform variations
    for _ in range(8):
        pts = np.random.rand(N, 2) * 0.8 + 0.1
        inits.append(np.concatenate([pts.flatten(), compute_safe_radii(pts)]))
        
    # Primary optimization phase
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
            cons_val = constraints(res.x)
            if np.min(cons_val) >= -1e-8:
                curr_sum = np.sum(res.x[2*N:])
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    # Secondary refinement phase: perturb best solution to escape local minima
    if best_vars is not None:
        for _ in range(12):
            x_pert = best_vars.copy()
            x_pert[:2*N] += np.random.randn(2*N) * 0.003
            x_pert[:2*N] = np.clip(x_pert[:2*N], 0.01, 0.99)
            x_pert[2*N:] = np.maximum(x_pert[2*N:] * 0.995, 1e-6)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
                cons_val = constraints(res.x)
                if np.min(cons_val) >= -1e-8:
                    curr_sum = np.sum(res.x[2*N:])
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
            except Exception:
                pass
                
        # Final high-precision polish
        res_final = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                             constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14})
        if np.min(constraints(res_final.x)) >= -1e-8:
            best_vars = res_final.x
            
    centers = best_vars[:2*N].reshape(N, 2)
    radii = np.maximum(best_vars[2*N:], 0.0)
    return centers, radii, float(np.sum(radii))
