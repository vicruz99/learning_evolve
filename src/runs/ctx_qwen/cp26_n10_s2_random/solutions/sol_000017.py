# sol_000017 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 38145db4) state=58c90071 sum of radii=2.621143 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_constraints(vars_flat):
    """Computes all boundary and non-overlap constraints for the packing."""
    X = vars_flat.reshape(N, 3)
    xs = X[:, 0]
    ys = X[:, 1]
    rs = X[:, 2]

    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c1 = xs - rs
    c2 = 1.0 - xs - rs
    c3 = ys - rs
    c4 = 1.0 - ys - rs

    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    idx = np.triu_indices(N, k=1)
    dx = xs[idx[0]] - xs[idx[1]]
    dy = ys[idx[0]] - ys[idx[1]]
    dr = rs[idx[0]] + rs[idx[1]]
    c5 = dx**2 + dy**2 - dr**2

    return np.concatenate([c1, c2, c3, c4, c5])

def compute_objective(vars_flat):
    """Objective: maximize sum of radii -> minimize negative sum."""
    rs = vars_flat[2::3]
    return -np.sum(rs)

def get_bounds():
    """Returns variable bounds for x, y, r."""
    b = []
    for _ in range(N):
        b.append((0.0, 1.0))
        b.append((0.0, 1.0))
        b.append((0.0, 0.5))
    return b

def generate_initial_guess(seed):
    """Generates a feasible initial configuration with small radii."""
    rng = np.random.default_rng(seed)
    pos = []
    # Base 5x5 grid with jitter
    for i in range(5):
        for j in range(5):
            pos.append([0.1 + 0.2*i + rng.uniform(-0.02, 0.02),
                        0.1 + 0.2*j + rng.uniform(-0.02, 0.02)])
    # Add extra circles in gaps
    pos.append([0.5 + rng.uniform(-0.05, 0.05), 0.05 + rng.uniform(-0.02, 0.02)])
    while len(pos) < N:
        pos.append([rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)])
    pos = pos[:N]

    init = np.zeros(N * 3)
    for i, p in enumerate(pos):
        init[3*i] = p[0]
        init[3*i+1] = p[1]
        init[3*i+2] = 0.04  # Small initial radius ensures feasibility
    return init

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}

    best_vars = None
    best_obj = -np.inf

    # Multiple restarts to escape local minima
    seeds = [0, 42, 123, 256, 999, 1000, 1337]
    for seed in seeds:
        x0 = generate_initial_guess(seed)
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 1500, 'ftol': 1e-9, 'disp': False})
            
            # Verify feasibility
            c_vals = compute_constraints(res.x)
            if np.all(c_vals >= -1e-7):
                curr_obj = -res.fun
                if curr_obj > best_obj:
                    best_obj = curr_obj
                    best_vars = res.x.copy()
        except Exception:
            pass

    # Fallback if optimization fails entirely
    if best_vars is None:
        best_vars = generate_initial_guess(0)

    centers = best_vars.reshape(N, 3)[:, :2]
    radii = best_vars.reshape(N, 3)[:, 2]
    
    # Ensure non-negative radii and numerical safety
    radii = np.maximum(radii, 0.0)
    sum_r = float(np.sum(radii))
    
    return centers, radii, sum_r
