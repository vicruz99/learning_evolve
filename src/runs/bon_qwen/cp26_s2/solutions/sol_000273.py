# sol_000273 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9068c8d6) state=89f381c4 sum of radii=2.616966 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(vars_):
    """Minimize negative sum of radii."""
    return -np.sum(vars_[2::3])

def _constraints(vars_):
    """Vectorized inequality constraints >= 0."""
    n = len(vars_) // 3
    x = vars_[0::3]
    y = vars_[1::3]
    r = vars_[2::3]
    
    # Pairwise non-overlap: dist^2 - (r_i + r_j)^2 >= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    c_ov = d2 - r_sum**2
    
    # Extract upper triangle (unique pairs)
    tri_i, tri_j = np.triu_indices(n, k=1)
    c_ov = c_ov[tri_i, tri_j]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_b = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    return np.concatenate([c_ov, c_b])

def _get_init(n, seed):
    """Generate feasible initial configuration from perturbed hex grid."""
    rng = np.random.default_rng(seed)
    rows = 6
    pts = []
    for i in range(rows):
        y = i * np.sqrt(3) / 2.0
        x0 = 0.5 if i % 2 == 1 else 0.0
        for j in range(rows):
            x = x0 + j
            pts.append([x, y])
    pts = np.array(pts[:n])
    
    # Normalize to [0, 1]
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    pts = (pts - pmin) / (pmax - pmin)
    
    # Add margin to keep circles strictly inside initially
    margin = 0.15
    pts = pts * (1.0 - 2.0 * margin) + margin
    pts += rng.normal(0, 0.005, size=pts.shape)
    
    # Consistent initial radii
    r_init = np.full(n, 0.06)
    
    x0 = np.zeros(3 * n)
    x0[0::3] = pts[:, 0]
    x0[1::3] = pts[:, 1]
    x0[2::3] = r_init
    return x0

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': _constraints}
    
    best_sum = 0.0
    best_vars = None
    
    # Multiple restarts to escape local minima
    for seed in range(8):
        x0 = _get_init(n, seed)
        try:
            res = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2500, 'ftol': 1e-12, 'disp': False})
            if res.success:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization unexpectedly fails
    if best_vars is None:
        best_vars = _get_init(n, 0)
        best_sum = np.sum(best_vars[2::3])
        
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    centers[:, 0] = best_vars[0::3]
    centers[:, 1] = best_vars[1::3]
    radii[:] = best_vars[2::3]
    
    return centers, radii, float(np.sum(radii))
