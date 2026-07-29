# sol_000110 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 82191eeb) state=35eb9720 sum of radii=2.500529 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _get_constraints(x):
    n = 26
    r = x[-1]
    centers = x[:2*n].reshape(n, 2)
    
    # Boundary constraints: centers must be within [r, 1-r]
    b = np.concatenate([
        centers[:, 0] - r,
        1.0 - centers[:, 0] - r,
        centers[:, 1] - r,
        1.0 - centers[:, 1] - r
    ])
    
    # Pairwise non-overlap constraints
    # Compute all pairwise distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Extract upper triangle (i < j)
    idx = np.triu_indices(n, k=1)
    p = dist[idx] - 2.0 * r
    
    return np.concatenate([b, p])

def _get_objective(x):
    return -x[-1]

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    
    # Initial configuration: Hexagonal-like arrangement
    r0 = 0.095
    dx = 2.0 * r0
    dy = np.sqrt(3.0) * r0
    
    coords = []
    y = r0 + 0.02
    x_base = r0 + 0.02
    
    # Pattern: 5, 4, 5, 4, 5, 3 circles per row (total 26)
    rows = [5, 4, 5, 4, 5, 3]
    for idx, count in enumerate(rows):
        offset = (dx / 2.0) if (idx % 2 == 1) else 0.0
        for i in range(count):
            coords.append([x_base + offset + i * dx, y])
        y += dy
        
    cx = np.array(coords)
    # Scale and shift to fit nicely in [0,1]^2
    cx = (cx - cx.min(axis=0)) * 0.85 / (cx.max(axis=0) - cx.min(axis=0)) + 0.075
    
    x0 = np.concatenate([cx.flatten(), [r0]])
    x0 += rng.normal(0, 0.0005, size=x0.shape)
    
    bounds = [(0.0, 1.0)] * (2*n) + [(0.08, 0.15)]
    cons = {'type': 'ineq', 'fun': _get_constraints}
    
    res = minimize(_get_objective, x0, method='SLSQP', bounds=bounds, 
                   constraints=cons, options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False})
                   
    if res.success:
        return res.x[:2*n].reshape(n, 2), np.full(n, res.x[-1]), res.x[-1] * n
    else:
        # Fallback to a valid grid packing if optimization fails
        grid = np.linspace(0.1, 0.9, 5)
        cx_f, cy_f = np.meshgrid(grid, grid)
        centers_f = np.vstack([cx_f.flatten(), cy_f.flatten()]).T
        centers_f = np.vstack([centers_f, [0.5, 0.5]])
        return centers_f, np.full(26, 0.08), 0.08 * 26
