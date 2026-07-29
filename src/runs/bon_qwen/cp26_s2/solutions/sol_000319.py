# sol_000319 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ef4a4e64) state=9b647417 sum of radii=2.614689 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(vars, N):
    """Negative sum of radii for minimization."""
    return -np.sum(vars[2*N:])

def _constraint_fun(vars, N):
    """Returns array of inequality constraints >= 0."""
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    # Boundary constraints: 0 <= x-r, x+r <= 1, 0 <= y-r, y+r <= 1
    c_boundary = np.concatenate([
        centers[:, 0] - radii,
        1 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1 - centers[:, 1] - radii
    ])
    
    # Overlap constraints: dist_ij >= r_i + r_j
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    i, j = np.triu_indices(N, k=1)
    c_overlap = dists[i, j] - radii[i] - radii[j]
    
    return np.concatenate([c_boundary, c_overlap])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    np.random.seed(42)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    cons = {'type': 'ineq', 'fun': lambda v: _constraint_fun(v, N)}
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-4, 0.5)] * N
    
    # Hexagonal initial guess: 5 rows, staggered columns (5,6,5,6,4)
    rows = 5
    dy = 1.0 / (rows + 1)
    y_base = np.linspace(dy, 1 - dy, rows)
    counts = [5, 6, 5, 6, 4]
    
    base_centers = []
    for i, n_cols in enumerate(counts):
        y = y_base[i]
        offset = 0.05 if i % 2 == 1 else 0.0
        if n_cols == 0: 
            continue
        margin = 0.05
        x_vals = np.linspace(margin + offset, 1.0 - margin + offset, n_cols)
        for x in x_vals:
            base_centers.append([x, y])
            
    base_centers = np.array(base_centers)
    if len(base_centers) != N:
        base_centers = np.random.rand(N, 2) * 0.8 + 0.1
        
    init_radii = np.full(N, 0.05)
    
    # Multiple restarts to escape local minima
    for _ in range(10):
        centers_start = base_centers.copy()
        centers_start += np.random.randn(N, 2) * 0.01
        centers_start = np.clip(centers_start, 0.02, 0.98)
        
        radii_start = init_radii + np.random.randn(N) * 0.005
        radii_start = np.clip(radii_start, 0.01, 0.15)
        
        x0 = np.concatenate([centers_start.flatten(), radii_start])
        
        try:
            res = minimize(_objective, x0, args=(N,), method='SLSQP', 
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                c_val = _constraint_fun(res.x, N)
                if np.min(c_val) >= -1e-6:
                    best_sum = curr_sum
                    best_centers = res.x[:2*N].reshape(N, 2)
                    best_radii = res.x[2*N:]
        except Exception:
            pass
            
    if best_centers is None:
        best_centers = base_centers
        best_radii = init_radii
        
    # Post-processing: resolve any remaining tiny overlaps strictly
    for _ in range(5):
        violated = False
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j])**2))
                if dist < best_radii[i] + best_radii[j] - 1e-9:
                    scale = (dist + 1e-9) / (best_radii[i] + best_radii[j])
                    best_radii[i] *= scale
                    best_radii[j] *= scale
                    violated = True
        if not violated:
            break
            
    # Final boundary clamping
    for i in range(N):
        r = best_radii[i]
        best_centers[i, 0] = np.clip(best_centers[i, 0], r, 1-r)
        best_centers[i, 1] = np.clip(best_centers[i, 1], r, 1-r)
        
    return best_centers, best_radii, float(np.sum(best_radii))
