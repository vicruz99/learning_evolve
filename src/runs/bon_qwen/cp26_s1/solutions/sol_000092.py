# sol_000092 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9fb5006a) state=0457f770 sum of radii=2.603444 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(v):
    n = 26
    C = v[:52].reshape(n, 2)
    R = v[52:]
    
    # Pairwise distances
    diff = C[:, None, :] - C[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Strict lower triangle indices for pairs
    i, j = np.tril_indices(n, k=-1)
    pair_con = dists[i, j] - (R[i] + R[j])
    
    # Wall constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    wall_con = np.concatenate([
        C[:, 0] - R,
        1.0 - C[:, 0] - R,
        C[:, 1] - R,
        1.0 - C[:, 1] - R
    ])
    
    return np.concatenate([pair_con, wall_con])

def obj(v):
    return -np.sum(v[52:])

def run_packing():
    n = 26
    best_sum = -np.inf
    best_v = None
    
    bounds = [(0.0, 1.0)] * 52 + [(1e-4, 0.5)] * 26
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    starts = []
    
    # 1. Feasible Hexagonal grid initialization
    centers_hex = np.zeros((n, 2))
    idx = 0
    row_counts = [4, 5, 4, 5, 4, 4]
    for r_idx, cnt in enumerate(row_counts):
        y = 0.12 + r_idx * 0.18
        for c_idx in range(cnt):
            x = 0.12 + c_idx * 0.20 + (0.10 if r_idx % 2 == 1 else 0)
            centers_hex[idx] = [x, y]
            idx += 1
    starts.append(np.concatenate([centers_hex.flatten(), np.full(n, 0.06)]))
    
    # 2. Perturbed hex grids
    for seed in range(3):
        rng = np.random.default_rng(seed + 100)
        p = centers_hex.copy()
        p += rng.normal(0, 0.02, (n, 2))
        p = np.clip(p, 0.05, 0.95)
        starts.append(np.concatenate([p.flatten(), np.full(n, 0.06)]))
        
    # 3. Random grid starts
    for seed in range(2):
        rng = np.random.default_rng(seed + 200)
        c = rng.uniform(0.15, 0.85, (n, 2))
        starts.append(np.concatenate([c.flatten(), np.full(n, 0.04)]))

    # Optimize from each start
    for v0 in starts:
        try:
            res = minimize(obj, v0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_v = res.x.copy()
        except Exception:
            continue
            
    if best_v is None:
        best_v = starts[0]
        
    centers = best_v[:52].reshape(n, 2)
    radii = best_v[52:]
    
    # Post-processing: enforce strict validity with safety margin
    for _ in range(50):
        diff = centers[:, None, :] - centers[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        i, j = np.tril_indices(n, k=-1)
        violations = (radii[i] + radii[j]) - dists[i, j]
        max_v = np.max(violations)
        
        if max_v > 1e-10:
            radii *= (1.0 - max_v * 0.01 - 1e-4)
            continue
            
        # Enforce boundary constraints strictly
        for k in range(n):
            x, y = centers[k]
            r = radii[k]
            min_w = min(x, 1.0 - x, y, 1.0 - y)
            if r > min_w + 1e-10:
                radii[k] = min_w - 1e-10
                
        break
        
    return centers, radii, float(np.sum(radii))
