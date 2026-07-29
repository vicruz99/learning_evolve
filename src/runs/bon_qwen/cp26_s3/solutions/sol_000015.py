# sol_000015 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0f0997f0) state=2aaed84b sum of radii=2.631094 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_initial_config():
    """Generates a feasible hexagonal initial configuration for 26 circles."""
    N = 26
    r0 = 0.08
    xs, ys = [], []
    row_height = np.sqrt(3) * r0
    counts = [5, 5, 5, 5, 6]
    y_base = 2 * r0
    
    for i, count in enumerate(counts):
        y = y_base + i * row_height
        x_start = 0.5 - (count - 1) * r0
        for k in range(count):
            xs.append(x_start + k * 2 * r0)
            ys.append(y)
            
    rs = np.full(N, r0)
    return np.array(xs), np.array(ys), rs

def boundary_constraints(vars):
    """Ensures circles remain within the unit square."""
    N = 26
    xs = vars[:N]
    ys = vars[N:2*N]
    rs = vars[2*N:]
    return np.concatenate([
        xs - rs,          # x >= r
        1.0 - xs - rs,    # x + r <= 1
        ys - rs,          # y >= r
        1.0 - ys - rs     # y + r <= 1
    ])

def distance_constraints(vars):
    """Ensures circles do not overlap."""
    N = 26
    xs = vars[:N]
    ys = vars[N:2*N]
    rs = vars[2*N:]
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dr = rs[:, None] + rs[None, :]
    
    dist_sq = dx**2 + dy**2
    req_sq = dr**2
    
    # Extract lower triangle (i > j) to avoid duplicates and self-comparison
    mask = np.tril(np.ones((N, N), dtype=bool), k=-1)
    return (dist_sq - req_sq)[mask]

def objective_fun(vars):
    """Negative sum of radii to be minimized."""
    N = 26
    return -np.sum(vars[2*N:])

def boundary_fun(vars):
    return boundary_constraints(vars)

def distance_fun(vars):
    return distance_constraints(vars)

def run_packing():
    N = 26
    xs, ys, rs = get_initial_config()
    x0 = np.concatenate([xs, ys, rs])
    
    # Bounds for variables: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    
    cons = [
        {'type': 'ineq', 'fun': boundary_fun},
        {'type': 'ineq', 'fun': distance_fun}
    ]
    
    options = {'maxiter': 3000, 'ftol': 1e-10, 'disp': False}
    
    res = minimize(objective_fun, x0, method='SLSQP', bounds=bounds, 
                   constraints=cons, options=options)
    
    if res.success:
        final_xs = res.x[:N]
        final_ys = res.x[N:2*N]
        final_rs = res.x[2*N:]
    else:
        final_xs, final_ys, final_rs = xs, ys, rs
        
    # Ensure non-negative radii
    final_rs = np.maximum(final_rs, 0.0)
    
    centers = np.column_stack([final_xs, final_ys])
    return centers, final_rs, np.sum(final_rs)
