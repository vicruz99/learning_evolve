# sol_000032 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000028 (state 1c5b6a86) state=535dc621 sum of radii=2.618067 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(v, n):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def _constraints(v, n, i_idx, j_idx):
    """Compute inequality constraints: boundaries and squared non-overlap."""
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    bc = np.concatenate([c[:,0]-r, 1-c[:,0]-r, c[:,1]-r, 1-c[:,1]-r])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    ci, cj = c[i_idx], c[j_idx]
    di, dj = r[i_idx], r[j_idx]
    d2 = np.sum((ci - cj)**2, axis=1)
    s2 = (di + dj)**2
    
    return np.concatenate([bc, d2 - s2])

def run_packing():
    """
    Optimizes packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    i_idx, j_idx = np.triu_indices(n, k=1)
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': _constraints, 'args': (n, i_idx, j_idx)}

    best_val = -1e9
    best_x = None

    # Multiple restarts from perturbed hexagonal lattices
    for seed in range(30):
        np.random.seed(seed * 13 + 7)
        
        # Generate hexagonal lattice initialization
        centers = np.zeros((n, 2))
        r0 = 0.085
        y = r0
        row = 0
        idx = 0
        while idx < n:
            x = r0 if row % 2 == 0 else 2 * r0
            while x <= 1 - r0 and idx < n:
                centers[idx] = [x, y]
                x += 2 * r0
                idx += 1
            y += np.sqrt(3) * r0
            row += 1

        # Add controlled noise to break symmetry and explore space
        centers += np.random.uniform(-0.01, 0.01, centers.shape)
        centers = np.clip(centers, 0.03, 0.97)
        
        # Start with feasible radii
        r_init = np.full(n, 0.05)
        x0 = np.concatenate([centers.flatten(), r_init])

        try:
            res = minimize(_objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons_dict, options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_val:
                best_val = -res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    # Fallback in case optimization fails entirely
    if best_x is None:
        centers = np.random.uniform(0.2, 0.8, (n, 2))
        radii = np.full(n, 0.02)
        return centers, radii, float(np.sum(radii))

    centers = best_x[:2*n].reshape(n, 2)
    radii = best_x[2*n:]

    # Safety adjustment to strictly satisfy validation tolerances
    radii *= 0.9999995
    centers = np.clip(centers, 1e-8, 1.0 - 1e-8)

    return centers, radii, float(np.sum(radii))
