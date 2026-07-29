# sol_000139 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fa534539) state=b6056bd8 sum of radii=2.586790 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint

N_CIRCLES = 26

def compute_objective(vars):
    return -np.sum(vars[2*N_CIRCLES:])

def compute_constraints(vars):
    n = N_CIRCLES
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con = np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])
    
    # Pairwise non-overlap constraints: dist >= r_i + r_j
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Extract lower triangular part (i < j)
    mask = np.tril(np.ones((n, n), dtype=bool), k=-1)
    con = np.concatenate([con, dists[mask] - r_sum[mask]])
    
    return con

def run_packing():
    n = N_CIRCLES
    np.random.seed(42)
    
    # Initial feasible configuration: 5x5 grid + 1 center circle
    base = np.linspace(0.15, 0.85, 5)
    centers = np.array([[x, y] for x in base for y in base])
    centers = np.vstack([centers, [0.5, 0.5]])
    
    # Small random perturbation breaks symmetry and helps escape local minima
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    radii = np.full(n, 0.08)
    
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds: centers in [0, 1], radii in [1e-6, 0.5]
    lb = np.zeros(3*n)
    ub = np.ones(3*n)
    ub[2*n:] = 0.5
    lb[2*n:] = 1e-6
    bounds = Bounds(lb, ub)
    
    nlcon = NonlinearConstraint(compute_constraints, 0, np.inf)
    
    res = minimize(compute_objective, x0, method='trust-constr',
                   bounds=bounds, constraints=nlcon,
                   options={'maxiter': 2000, 'verbose': 0})
    
    c_opt = res.x[:2*n].reshape(n, 2)
    r_opt = res.x[2*n:]
    
    # Ensure non-negativity and apply microscopic shrink for numerical safety
    r_opt = np.maximum(r_opt, 1e-6) * 0.999999
    
    return c_opt, r_opt, np.sum(r_opt)
