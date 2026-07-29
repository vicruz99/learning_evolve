# sol_000321 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f00b2e18) state=aaecf92d sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty(vars, n, lam):
    centers = vars[:2*n].reshape(n, 2)
    radii = vars[2*n:]
    
    # Boundary penalties
    pen = np.sum(np.maximum(0, radii - centers[:, 0])**2)
    pen += np.sum(np.maximum(0, radii + centers[:, 0] - 1)**2)
    pen += np.sum(np.maximum(0, radii - centers[:, 1])**2)
    pen += np.sum(np.maximum(0, radii + centers[:, 1] - 1)**2)
    
    # Pairwise overlap penalties
    dx = centers[:, 0, None] - centers[:, None, 0]
    dy = centers[:, 1, None] - centers[:, None, 1]
    dists = np.sqrt(dx**2 + dy**2)
    rad_sum = radii[:, None] + radii[None, :]
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    violations = rad_sum - dists
    violations[~mask] = 0
    pen += np.sum(np.maximum(0, violations)**2)
    
    return lam * pen

def objective(vars, n, lam):
    return -np.sum(vars[2*n:]) + compute_penalty(vars, n, lam)

def solve_radii_exact(centers):
    n = centers.shape[0]
    # Initialize with boundary limits
    radii = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                       np.minimum(centers[:, 1], 1 - centers[:, 1]))
    
    dx = centers[:, 0, None] - centers[:, None, 0]
    dy = centers[:, 1, None] - centers[:, None, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    # Iteratively enforce pairwise constraints r_i + r_j <= d_ij
    for _ in range(100):
        diffs = dists - radii[None, :]
        diffs[np.eye(n, dtype=bool)] = np.inf
        new_radii = np.minimum(radii, np.min(diffs, axis=1))
        new_radii = np.maximum(new_radii, 0)
        if np.allclose(radii, new_radii, atol=1e-12):
            break
        radii = new_radii
    return radii

def run_packing():
    n = 26
    np.random.seed(42)
    
    # Hexagonal lattice initialization
    pts = []
    y = 0.08
    row = 0
    while len(pts) < n:
        x = 0.08
        shift = 0.04 if row % 2 == 1 else 0.0
        while x <= 0.92 and len(pts) < n:
            pts.append([x + shift, y])
            x += 0.1
        y += 0.0866025  # sqrt(3)/2 * 0.1
        row += 1
    pts = np.array(pts[:n])
    
    r_init = np.full(n, 0.08)
    x0 = np.concatenate([pts.ravel(), r_init])
    
    bounds = [(0, 1)]*(2*n) + [(0, 0.5)]*n
    
    lam = 10.0
    x_curr = x0
    for _ in range(7):
        lam *= 10
        res = minimize(objective, x_curr, args=(n, lam), method='L-BFGS-B', 
                       bounds=bounds, options={'ftol': 1e-15, 'gtol': 1e-13, 'maxiter': 3000})
        x_curr = res.x
        
    centers_opt = x_curr[:2*n].reshape(n, 2)
    radii_final = solve_radii_exact(centers_opt)
    
    # Tiny margin to strictly satisfy 1e-12 validation tolerance
    radii_final *= 0.999999999999
    
    return centers_opt, radii_final, np.sum(radii_final)
