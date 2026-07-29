# sol_000081 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000001 (state 1501c8b5) state=38c6cfa4 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective function: maximize sum of radii -> minimize negative sum."""
    return -np.sum(v[2*N:])

def constraints(v):
    """
    Computes all inequality constraints that must be >= 0:
    1. Boundary constraints: circles inside [0,1]^2
    2. Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    """
    x = v[:N]
    y = v[N:2*N]
    r = v[2*N:]
    
    # Boundary constraints
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Vectorized pairwise squared distance constraints
    X = x[:, None]
    Y = y[:, None]
    R = r[:, None]
    
    dx = X - X.T
    dy = Y - Y.T
    dr = R + R.T
    
    # Only need upper triangle for unique pairs
    idx = np.triu_indices(N, k=1)
    c = np.concatenate([c, dx[idx]**2 + dy[idx]**2 - dr[idx]**2])
    
    return c

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    best_val = -np.inf
    best_v = None
    
    # Multiple diverse starts to avoid local minima
    for seed in range(25):
        np.random.seed(seed)
        centers = np.zeros((N, 2))
        r_init = 0.075
        
        # Hexagonal staggered initialization
        y = r_init
        row = 0
        cnt = 0
        # Row lengths sum to 26
        row_lens = [6, 5, 6, 5, 4]
        
        for rl in row_lens:
            x_start = r_init if row % 2 == 0 else 2 * r_init
            for c_idx in range(rl):
                x = x_start + c_idx * 2 * r_init
                if x > 1 - r_init:
                    break
                centers[cnt] = [x, y]
                cnt += 1
                if cnt >= N:
                    break
            y += r_init * np.sqrt(3)
            row += 1
            if cnt >= N:
                break
                
        # Fill any missing slots randomly if lattice cutoff happened
        while cnt < N:
            centers[cnt] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
            cnt += 1
            
        centers = np.array(centers[:N])
        centers += np.random.uniform(-0.005, 0.005, centers.shape)
        centers = np.clip(centers, 0.02, 0.98)
        
        v0 = np.concatenate([centers.flatten(), np.full(N, r_init)])
        
        try:
            res = minimize(objective, v0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_val:
                best_val = -res.fun
                best_v = res.x.copy()
        except Exception:
            continue
            
        # Perturb best solution found so far to escape local minima
        if best_v is not None and seed > 5:
            v_pert = best_v + np.random.normal(0, 0.0008, size=best_v.shape)
            v_pert[2*N:] = np.maximum(v_pert[2*N:], 0.01)
            try:
                res = minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                               constraints={'type': 'ineq', 'fun': constraints},
                               options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                if -res.fun > best_val:
                    best_val = -res.fun
                    best_v = res.x.copy()
            except Exception:
                pass
                
    centers = best_v[:2*N].reshape(N, 2)
    radii = best_v[2*N:]
    
    # Negligible shrink to guarantee strict validity against float precision drift
    radii *= (1.0 - 1e-9)
    centers = np.clip(centers, 1e-9, 1.0 - 1e-9)
    
    # Iterative enforcement to strictly satisfy all validator tolerances
    for _ in range(3):
        for i in range(N):
            mr = radii[i]
            # Wall constraints
            mr = min(mr, centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
            # Neighbor constraints
            for j in range(N):
                if i == j:
                    continue
                d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
                mr = min(mr, d - radii[j])
            radii[i] = max(0.0, mr)
            
    return centers, radii, float(np.sum(radii))
