# sol_000189 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 624944be) state=f833e12e sum of radii=2.376970 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def evaluate_constraints(vars):
    n = 26
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    diffs_x = x[:, np.newaxis] - x[np.newaxis, :]
    diffs_y = y[:, np.newaxis] - y[np.newaxis, :]
    diffs_r = r[:, np.newaxis] + r[np.newaxis, :]
    
    dists_sq = diffs_x**2 + diffs_y**2
    overlaps = dists_sq - diffs_r**2
    
    # Extract upper triangle for i < j
    c = np.concatenate([c, np.triu(overlaps, k=1).flatten()])
    return c

def objective(vars):
    r = vars[2::3]
    return -np.sum(r)

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Initial configuration: 5x5 grid + 1 center
    xs = np.linspace(0.125, 0.875, 5)
    ys = np.linspace(0.125, 0.875, 5)
    xx, yy = np.meshgrid(xs, ys)
    centers = np.vstack([xx.ravel(), yy.ravel()]).T
    centers = np.vstack([centers, [0.5, 0.5]])
    
    # Perturb to break symmetry and avoid flat gradients
    centers += np.random.uniform(-0.005, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    r_init = 0.04
    radii_init = np.full(n, r_init)
    
    # Interleave variables: x0, y0, r0, x1, y1, r1, ...
    x0 = np.empty(3*n)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii_init
    
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    constraints = [{'type': 'ineq', 'fun': evaluate_constraints}]
    
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        options={'maxiter': 5000, 'ftol': 1e-12}
    )
    
    if res.success:
        best_x = res.x[0::3]
        best_y = res.x[1::3]
        best_r = res.x[2::3]
    else:
        best_x = centers[:, 0]
        best_y = centers[:, 1]
        best_r = radii_init
        
    # Ensure strictly positive radii
    best_r = np.maximum(best_r, 1e-6)
    best_centers = np.column_stack([best_x, best_y])
    
    return best_centers, best_r, np.sum(best_r)
