# sol_000091 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e9cb3956) state=98def28e sum of radii=2.617869 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_penalty_and_obj(vars, beta):
    """
    Computes the penalized objective value and the raw penalty sum.
    vars: flattened array [x1, y1, r1, x2, y2, r2, ...]
    beta: penalty weight
    """
    n = 26
    xs = vars[0::3]
    ys = vars[1::3]
    rs = vars[2::3]
    
    # Boundary penalties (squared violations)
    pen = np.sum(np.maximum(0.0, rs - xs)**2)
    pen += np.sum(np.maximum(0.0, xs + rs - 1.0)**2)
    pen += np.sum(np.maximum(0.0, rs - ys)**2)
    pen += np.sum(np.maximum(0.0, ys + rs - 1.0)**2)
    
    # Overlap penalties (vectorized)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    overlaps = rs[:, None] + rs[None, :] - dist
    np.fill_diagonal(overlaps, 0.0)
    pen += np.sum(np.maximum(0.0, overlaps)**2)
    
    obj = -np.sum(rs) + beta * pen
    return obj, pen

def objective(vars, beta):
    obj, _ = compute_penalty_and_obj(vars, beta)
    return obj

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    
    # Initial feasible configuration: 5x5 grid + center
    x_grid = np.repeat(np.linspace(0.1, 0.9, 5), 5)
    y_grid = np.tile(np.linspace(0.1, 0.9, 5), 5)
    xs = np.append(x_grid, 0.5)
    ys = np.append(y_grid, 0.5)
    rs = np.full(n, 0.09)
    
    # Add small random noise to break symmetry and avoid grid stagnation
    xs += rng.uniform(-0.005, 0.005, n)
    ys += rng.uniform(-0.005, 0.005, n)
    xs = np.clip(xs, 0.02, 0.98)
    ys = np.clip(ys, 0.02, 0.98)
    
    vars_init = np.zeros(3 * n)
    vars_init[0::3] = xs
    vars_init[1::3] = ys
    vars_init[2::3] = rs
    
    # Bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0) if i % 3 != 2 else (0.0, 0.5) for i in range(3 * n)]
    
    best_vars = vars_init
    # Anneal penalty weight to gradually enforce constraints strictly
    betas = [50.0, 500.0, 5000.0, 50000.0, 200000.0]
    
    for beta in betas:
        res = minimize(objective, best_vars, args=(beta,), method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 4000, 'ftol': 1e-13, 'gtol': 1e-13})
        best_vars = res.x
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    
    # Final robustness check: if numerical noise left tiny violations, scale down slightly
    _, final_pen = compute_penalty_and_obj(best_vars, 1e10)
    if final_pen > 1e-12:
        # Conservative scaling to guarantee validator passes
        scale_factor = 1.0 - 2.0 * np.sqrt(final_pen) / (np.max(radii) + 1e-9)
        radii = radii * max(scale_factor, 0.999)
        centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
        
    return centers, radii, np.sum(radii)
