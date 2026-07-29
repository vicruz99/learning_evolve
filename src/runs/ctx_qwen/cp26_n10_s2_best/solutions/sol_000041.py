# sol_000041 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000016 (state 585439f0) state=d8beba0e sum of radii=2.593462 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2*N:])

def constraint_func(vars, i_idx, j_idx):
    """Inequality constraints: boundaries and non-overlap."""
    x = vars[:N]
    y = vars[N:2*N]
    r = vars[2*N:]
    
    num_pair = len(i_idx)
    c = np.empty(4*N + num_pair)
    idx = 0
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c[idx:idx+N] = x - r; idx += N
    c[idx:idx+N] = 1.0 - x - r; idx += N
    c[idx:idx+N] = y - r; idx += N
    c[idx:idx+N] = 1.0 - y - r; idx += N
    
    # Pairwise non-overlap constraints: dist >= r_i + r_j
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dists = np.sqrt(dx*dx + dy*dy)
    r_sums = r[i_idx] + r[j_idx]
    c[idx:] = dists - r_sums
    
    return c

def run_packing():
    i_idx, j_idx = np.triu_indices(N, k=1)
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraint_func, 'args': (i_idx, j_idx)}
    
    best_val = -1.0
    best_x = None
    np.random.seed(42)
    
    # Multi-start optimization with varied initializations
    for trial in range(15):
        if trial < 10:
            # Hexagonal lattice variations
            rows = [6, 5, 6, 5, 4] if trial % 2 == 0 else [5, 6, 5, 6, 4]
            x0, y0 = [], []
            r_base = 0.08 + 0.005 * (trial % 3)
            y = r_base
            row = 0
            for count in rows:
                x_start = r_base if row % 2 == 0 else 2*r_base
                x = x_start
                for _ in range(count):
                    if len(x0) < N:
                        x0.append(x)
                        y0.append(y)
                        x += 2*r_base
                y += r_base * np.sqrt(3)
                row += 1
        else:
            # Random dense initialization
            x0 = np.random.uniform(0.05, 0.95, N)
            y0 = np.random.uniform(0.05, 0.95, N)
            
        # Add controlled jitter to break symmetry
        jitter = np.random.uniform(-0.015, 0.015, (N, 2))
        centers = np.column_stack((x0[:N], y0[:N])) + jitter
        centers = np.clip(centers, 0.05, 0.95)
        
        r0 = np.full(N, 0.03)  # Safe initial radius
        x_init = np.concatenate([centers.ravel(), r0])
        
        try:
            res = minimize(objective, x_init, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            
            if -res.fun > best_val:
                slacks = constraint_func(res.x, i_idx, j_idx)
                if np.all(slacks >= -1e-7):
                    best_val = -res.fun
                    best_x = res.x.copy()
        except Exception:
            pass
            
    if best_x is None:
        # Fallback to a safe random configuration
        best_x = np.concatenate([np.random.uniform(0.1, 0.9, 2*N), np.full(N, 0.03)])
        
    centers = np.column_stack((best_x[:N], best_x[N:2*N]))
    radii = best_x[2*N:]
    
    # Post-processing: Strictly enforce boundary constraints
    for i in range(N):
        radii[i] = min(radii[i], centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        
    # Post-processing: Iteratively resolve pairwise overlaps to meet 1e-12 tolerance
    for _ in range(3):
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                if radii[i] + radii[j] > d + 1e-13:
                    delta = (radii[i] + radii[j] - d) / 2.0 + 1e-14
                    radii[i] = max(0.0, radii[i] - delta)
                    radii[j] = max(0.0, radii[j] - delta)
                    
    return centers, radii, float(np.sum(radii))
