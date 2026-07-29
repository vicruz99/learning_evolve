# sol_000038 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000021 (state 2060a481) state=25e0cc5d sum of radii=2.603867 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(x):
    """Minimize negative sum of radii (indices 2, 5, 8, ...)"""
    return -np.sum(x[2::3])

def constraint_fun(x):
    """Evaluate all inequality constraints: boundary and non-overlap (squared)."""
    cs = x.reshape(N, 3)
    xs, ys, rs = cs[:, 0], cs[:, 1], cs[:, 2]
    
    # Preallocate constraint array
    m = N * (N - 1) // 2
    cons = np.empty(4 * N + m)
    idx = 0
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    cons[idx:idx+N] = xs - rs; idx += N
    cons[idx:idx+N] = 1.0 - xs - rs; idx += N
    cons[idx:idx+N] = ys - rs; idx += N
    cons[idx:idx+N] = 1.0 - ys - rs; idx += N
    
    # Overlap constraints: (dx^2 + dy^2) - (ri + rj)^2 >= 0
    # Vectorized computation for all pairs i < j
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    triu_idx = np.triu_indices(N, k=1)
    cons[idx:] = dx[triu_idx]**2 + dy[triu_idx]**2 - dr[triu_idx]**2
    
    return cons

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N
    best_sum = -1.0
    best_x = None
    
    # Variable bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1), (0, 1), (0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    # Phase 1: Multiple diverse initializations
    for seed in range(25):
        np.random.seed(seed)
        pts = []
        # Hexagonal lattice approximation for dense packing
        r_est = 0.095
        for row in range(15):
            y = r_est + row * np.sqrt(3) * r_est
            if y > 1 - r_est: break
            for col in range(15):
                shift = r_est if row % 2 == 1 else 0
                x = r_est + shift + col * 2 * r_est
                if x > 1 - r_est: break
                pts.append([x, y])
                
        if len(pts) < n:
            for i in range(10):
                for j in range(10):
                    pts.append([0.05 + i*0.1, 0.05 + j*0.1])
                    
        np.random.shuffle(pts)
        pts = pts[:n]
        
        # Flatten to optimization vector [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = pts[i][0]
            x0[3*i+1] = pts[i][1]
            x0[3*i+2] = 0.04  # Start small to guarantee feasibility
            
        # Add noise to break symmetry and avoid grid-locking
        x0 += np.random.normal(0, 0.005, 3*n)
        for i in range(n):
            x0[3*i] = np.clip(x0[3*i], 0.05, 0.95)
            x0[3*i+1] = np.clip(x0[3*i+1], 0.05, 0.95)
            x0[3*i+2] = max(0.01, x0[3*i+2])
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                c = constraint_fun(res.x)
                if np.all(c >= -1e-7):
                    best_sum = -res.fun
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local refinement of the best found configuration
    if best_x is not None:
        for _ in range(10):
            x0 = best_x + np.random.normal(0, 1e-4, 3*n)
            for i in range(n):
                x0[3*i] = np.clip(x0[3*i], 0, 1)
                x0[3*i+1] = np.clip(x0[3*i+1], 0, 1)
                x0[3*i+2] = max(0, x0[3*i+2])
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                if -res.fun > best_sum:
                    c = constraint_fun(res.x)
                    if np.all(c >= -1e-7):
                        best_sum = -res.fun
                        best_x = res.x.copy()
            except Exception:
                pass
                
    # Fallback safety net
    if best_x is None:
        x0 = np.zeros(3*n)
        for i in range(n):
            x0[3*i] = 0.5
            x0[3*i+1] = 0.5
            x0[3*i+2] = 0.05
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            best_x = res.x
            best_sum = -res.fun
        except Exception:
            pass
        
    # Extract and format results
    centers = best_x.reshape(n, 3)[:, :2]
    radii = best_x.reshape(n, 3)[:, 2]
    return centers, radii, float(np.sum(radii))
