# sol_000051 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state baeb2167) state=12cee48d sum of radii=2.600217 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(centers_flat, radii_flat):
    """
    Compute inequality constraints for the circle packing problem.
    Constraints are:
    1. x - r >= 0
    2. 1 - x - r >= 0
    3. y - r >= 0
    4. 1 - y - r >= 0
    5. dist(i,j) - r_i - r_j >= 0 for all i < j
    Returns an array where all elements must be >= 0.
    """
    n = len(radii_flat)
    c = centers_flat.reshape(n, 2)
    r = radii_flat
    
    cons = []
    # Boundary constraints
    cons.extend(c[:, 0] - r)
    cons.extend(1.0 - c[:, 0] - r)
    cons.extend(c[:, 1] - r)
    cons.extend(1.0 - c[:, 1] - r)
    
    # Overlap constraints (vectorized)
    diffs = c[:, None, :] - c[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    idx = np.triu_indices(n, k=1)
    cons.extend(dists[idx] - r[idx[0]] - r[idx[1]])
    
    return np.array(cons)

def _obj_26(x):
    """Objective function: minimize negative sum of radii."""
    return -x[52:].sum()

def _cons_26(x):
    """Constraint function for N=26."""
    return compute_constraints(x[:52], x[52:])

def run_packing():
    N = 26
    # Bounds: centers in [0, 1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N

    # Initial configuration: structured grid with perturbation
    xs = np.linspace(0.15, 0.85, 6)
    ys = np.linspace(0.15, 0.85, 5)
    pts = []
    for y in ys:
        for x in xs:
            pts.append([x, y])
    pts = np.array(pts[:N])
    pts += np.random.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.01, 0.99)
    r_init = np.full(N, 0.05)
    x0 = np.concatenate([pts.ravel(), r_init])

    best_x = x0.copy()
    best_sum_r = -np.inf

    # Iterative optimization with restarts
    for iteration in range(5):
        try:
            res = minimize(_obj_26, x0, method='SLSQP', bounds=bounds,
                          constraints={'type': 'ineq', 'fun': _cons_26},
                          options={'maxiter': 1500, 'ftol': 1e-9})
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum_r:
                    best_sum_r = current_sum
                    best_x = res.x
            
            # Perturb best solution for next iteration to escape local minima
            noise = np.random.uniform(-0.005, 0.005, best_x.shape)
            x0 = best_x + noise
            for i, b in enumerate(bounds):
                x0[i] = np.clip(x0[i], b[0], b[1])
        except Exception:
            continue

    centers = best_x[:2*N].reshape(N, 2)
    radii = best_x[2*N:]
    return centers, radii, radii.sum()
