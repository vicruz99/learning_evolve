# sol_000040 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 12653929) state=76526f9c sum of radii=2.587002 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

N = 26

def objective_func(v):
    # Minimize negative sum of radii
    return -np.sum(v[2::3])

def constraint_boundary(v):
    # 4N constraints: x-r>=0, 1-x-r>=0, y-r>=0, 1-y-r>=0
    v = v.reshape(N, 3)
    x, y, r = v[:, 0], v[:, 1], v[:, 2]
    return np.concatenate([x - r, 1 - x - r, y - r, 1 - y - r])

def constraint_overlap(v):
    # N*(N-1)/2 constraints: dist_ij - (r_i + r_j) >= 0
    v = v.reshape(N, 3)
    centers = v[:, :2]
    r = v[:, 2]
    dists = squareform(pdist(centers))
    r_sum = r[:, None] + r[None, :]
    diff = dists - r_sum
    return diff[np.triu_indices(N, k=1)]

def get_initial_config(seed):
    rng = np.random.RandomState(seed)
    # Hexagonal grid initialization for dense packing
    s = 0.18
    centers = []
    r_init = s / 4.5
    y = s / 2
    while y < 1 - s / 2 and len(centers) < N:
        x = s / 2
        shift = 0 if int(y / (s * np.sqrt(3) / 2)) % 2 == 0 else s / 2
        while x < 1 - s / 2 and len(centers) < N:
            centers.append([x + shift, y])
            x += s
        y += s * np.sqrt(3) / 2
    
    # Fill remaining if any (shouldn't happen with s=0.18 for N=26)
    while len(centers) < N:
        centers.append(rng.rand(2))
        
    centers = np.array(centers)
    radii = np.full(N, r_init)
    return np.hstack([centers.ravel(), radii])

def run_packing():
    best_v = None
    best_val = -np.inf
    bounds = [(0, 1) for _ in range(2 * N)] + [(0, 0.5) for _ in range(N)]
    cons = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # Try multiple random starts to avoid poor local minima
    for seed in range(5):
        x0 = get_initial_config(seed)
        try:
            res = minimize(objective_func, x0, bounds=bounds, constraints=cons, 
                           method='SLSQP', options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_val:
                best_val = -res.fun
                best_v = res.x
        except Exception:
            pass
            
    if best_v is not None:
        centers = best_v.reshape(N, 3)[:, :2]
        radii = best_v.reshape(N, 3)[:, 2]
        # Safety clipping to ensure strict validity
        radii = np.maximum(radii, 0.0)
        centers = np.clip(centers, 0.0, 1.0)
        return centers, radii, best_val
    else:
        # Fallback valid packing
        centers = np.random.rand(N, 2) * 0.8 + 0.1
        radii = np.full(N, 0.01)
        return centers, radii, 0.26
