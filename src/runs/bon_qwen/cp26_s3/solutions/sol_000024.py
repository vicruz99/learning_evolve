# sol_000024 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfc1b343) state=19c2e895 sum of radii=2.593482 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def boundary_constrs(x):
    """Returns boundary constraint violations as an array."""
    r = x[:N]
    c = x[N:].reshape((N, 2))
    return np.concatenate([
        c[:, 0] - r,
        1 - c[:, 0] - r,
        c[:, 1] - r,
        1 - c[:, 1] - r
    ])

def pair_constrs(x):
    """Returns pairwise non-overlap constraint violations as an array."""
    r = x[:N]
    c = x[N:].reshape((N, 2))
    cons = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            cons.append(np.sqrt(dx * dx + dy * dy) - (r[i] + r[j]))
    return np.array(cons)

def run_packing():
    np.random.seed(42)
    
    # Initial configuration: 5x5 grid + center circle
    centers = np.zeros((N, 2))
    radii = np.full(N, 0.08)
    for i in range(N):
        row = i // 5
        col = i % 5
        centers[i, 0] = 0.1 + col * 0.2
        centers[i, 1] = 0.1 + row * 0.2
    centers[25, 0] = 0.5
    centers[25, 1] = 0.5
    
    # Small perturbation to break symmetry and avoid flat local optima
    centers += np.random.uniform(-0.005, 0.005, size=centers.shape)
    radii = np.clip(radii, 0.01, 0.15)
    centers = np.clip(centers, 0.02, 0.98)
    
    def objective(x):
        return -np.sum(x[:N])
        
    x0 = np.concatenate([radii, centers.flatten()])
    bounds = [(0, None)] * N + [(0, 1)] * (2 * N)
    
    constraints = [
        {'type': 'ineq', 'fun': boundary_constrs},
        {'type': 'ineq', 'fun': pair_constrs}
    ]
    
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                   constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
                   
    opt_r = res.x[:N]
    opt_c = res.x[N:].reshape((N, 2))
    
    # Ensure non-negativity and strict feasibility for the validator
    opt_r = np.maximum(opt_r, 0.0)
    opt_r *= 0.99999
    
    return opt_c, opt_r, float(np.sum(opt_r))
