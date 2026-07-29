# sol_000051 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000016 (state 585439f0) state=0244174f sum of radii=2.617919 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(x):
    """Objective: minimize negative sum of radii (x[2N:])"""
    return -np.sum(x[2*N:])

def constraints(x):
    """Inequality constraints: boundaries and pairwise non-overlap"""
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:]
    
    # Pre-allocate constraint array
    m = 4 * N + N * (N - 1) // 2
    c = np.empty(m)
    idx = 0
    
    # Boundary constraints
    c[idx:idx+N] = cx - r; idx += N
    c[idx:idx+N] = 1.0 - cx - r; idx += N
    c[idx:idx+N] = cy - r; idx += N
    c[idx:idx+N] = 1.0 - cy - r; idx += N
    
    # Pairwise non-overlap constraints (vectorized)
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    c[idx:] = dist[i_idx, j_idx] - r_sum[i_idx, j_idx]
    
    return c

def run_packing():
    best_sum = 0.0
    best_x = None
    
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    np.random.seed(42)
    
    for seed in range(40):
        # Generate hexagonal lattice initialization
        r_init = 0.085 + np.random.uniform(0, 0.03)
        pts = []
        y = r_init
        row = 0
        while len(pts) < N + 15:
            x_start = r_init if row % 2 == 0 else 2 * r_init
            x = x_start
            while x <= 1 - r_init:
                pts.append([x, y])
                x += 2 * r_init
            y += r_init * np.sqrt(3)
            row += 1
            
        pts = np.array(pts)
        np.random.shuffle(pts)
        centers = pts[:N]
        
        # Add controlled jitter to break symmetry and explore space
        centers += np.random.uniform(-0.015, 0.015, size=centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        # Initial variable vector: [x_coords, y_coords, radii]
        x0 = np.concatenate([centers[:, 0], centers[:, 1], np.full(N, 0.04)])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            
            # Verify feasibility
            c_val = constraints(res.x)
            if np.all(c_val >= -1e-8):
                cur_sum = -res.fun
                if cur_sum > best_sum:
                    best_sum = cur_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Fallback (should not be reached with valid logic)
    if best_x is None:
        centers = np.tile([[0.5, 0.5]], (N, 1))
        radii = np.zeros(N)
        return centers, radii, 0.0
        
    cx = best_x[:N]
    cy = best_x[N:2*N]
    r = best_x[2*N:]
    
    centers = np.column_stack([cx, cy])
    radii = r.copy()
    
    # Ensure non-negative radii and valid bounds
    radii = np.maximum(0.0, radii)
    centers = np.clip(centers, 0.0, 1.0)
    
    # Strict feasibility adjustment to satisfy 1e-12 tolerance
    c_vals = constraints(np.concatenate([centers[:,0], centers[:,1], radii]))
    min_slack = np.min(c_vals)
    if min_slack < 0:
        max_r = np.max(radii)
        if max_r > 0:
            # Scale down radii proportionally to exactly satisfy the tightest constraint
            radii *= (1.0 + min_slack / max_r)
            
    return centers, radii, float(np.sum(radii))
