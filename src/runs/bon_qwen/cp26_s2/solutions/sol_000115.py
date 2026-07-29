# sol_000115 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 028484b6) state=591390e6 sum of radii=1.237364 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def overlap_constraint(vars):
    """Returns array of squared distance violations: dist^2 - (2r)^2 >= 0"""
    c = vars[:2*N].reshape((N, 2))
    r = vars[2*N]
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    idx = np.tril_indices(N, -1)
    return dist_sq[idx] - 4.0 * r**2

def boundary_constraint(vars):
    """Returns array of boundary violations: coord - r >= 0 and 1 - (coord + r) >= 0"""
    c = vars[:2*N].reshape((N, 2))
    r = vars[2*N]
    return np.concatenate([
        c[:, 0] - r,
        1.0 - (c[:, 0] + r),
        c[:, 1] - r,
        1.0 - (c[:, 1] + r)
    ])
    
def neg_r(vars):
    """Objective: maximize r => minimize -r"""
    return -vars[2*N]

def run_packing():
    np.random.seed(42)
    best_sol = None
    best_val = -1.0
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.05, 0.2)]
    constraints = [
        {'type': 'ineq', 'fun': overlap_constraint},
        {'type': 'ineq', 'fun': boundary_constraint}
    ]
    
    # Try multiple restarts to avoid local minima
    for _ in range(4):
        centers = np.empty((N, 2))
        idx = 0
        # Initialize with 5x5 grid
        for i in range(5):
            for j in range(5):
                if idx < N:
                    centers[idx] = [0.1 + 0.2 * j, 0.1 + 0.2 * i]
                    idx += 1
        if idx < N:
            centers[idx] = [0.5, 0.5]
            
        # Perturb to break symmetry
        centers += np.random.uniform(-0.015, 0.015, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        x0 = np.concatenate([centers.flatten(), [0.09]])
        
        try:
            res = minimize(
                neg_r,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
            )
            if res.success and res.x[2*N] > best_val:
                best_val = res.x[2*N]
                best_sol = res.x
        except Exception:
            continue
            
    if best_sol is None:
        # Fallback to simple grid packing
        centers = np.empty((N, 2))
        for i in range(5):
            for j in range(5):
                if i*5+j < N:
                    centers[i*5+j] = [0.1 + 0.2*j, 0.1 + 0.2*i]
        return centers, np.full(N, 0.09), 0.09*N
        
    centers = best_sol[:2*N].reshape((N, 2))
    r_opt = best_sol[2*N]
    
    # Compute safe radius to guarantee validity within tolerance
    min_dist = 1.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < min_dist: min_dist = d
        for coord in centers[i]:
            if coord < min_dist: min_dist = coord
            if 1.0 - coord < min_dist: min_dist = 1.0 - coord
            
    r_safe = min(r_opt, min_dist / 2.0)
    return centers, np.full(N, r_safe), r_safe * N
