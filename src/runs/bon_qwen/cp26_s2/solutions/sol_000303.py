# sol_000303 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 525683f8) state=cb9a0a15 sum of radii=2.593689 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Minimize negative sum of radii"""
    radii = vars[2::3]
    return -np.sum(radii)

def constraint_fun(vars):
    """
    Compute all inequality constraints.
    Returns array where each element >= 0 means constraint satisfied.
    Order: Wall constraints (4 per circle), then Pairwise distance constraints.
    """
    n_cons = 4 * N_CIRCLES + N_CIRCLES * (N_CIRCLES - 1) // 2
    c = np.empty(n_cons)
    idx = 0
    
    # Wall constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    for i in range(N_CIRCLES):
        x = vars[3 * i]
        y = vars[3 * i + 1]
        r = vars[3 * i + 2]
        c[idx] = x - r
        idx += 1
        c[idx] = 1.0 - x - r
        idx += 1
        c[idx] = y - r
        idx += 1
        c[idx] = 1.0 - y - r
        idx += 1
        
    # Pairwise constraints: dist(i,j) - r_i - r_j >= 0
    for i in range(N_CIRCLES):
        xi = vars[3 * i]
        yi = vars[3 * i + 1]
        ri = vars[3 * i + 2]
        for j in range(i + 1, N_CIRCLES):
            xj = vars[3 * j]
            yj = vars[3 * j + 1]
            rj = vars[3 * j + 2]
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx * dx + dy * dy)
            c[idx] = dist - ri - rj
            idx += 1
            
    return c

def get_bounds():
    """Bounds for [x, y, r] of each circle"""
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.append((0.0, 1.0))      # x in [0, 1]
        bounds.append((0.0, 1.0))      # y in [0, 1]
        bounds.append((1e-6, 0.5))     # r in [epsilon, 0.5]
    return bounds

def init_grid():
    """Structured initial guess: 5x5 grid + 1 center circle"""
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 5)
    gx, gy = np.meshgrid(xs, ys)
    
    centers_x = np.append(gx.flatten(), 0.5)
    centers_y = np.append(gy.flatten(), 0.5)
    radii = np.full(N_CIRCLES, 0.05)
    
    vars = np.empty(3 * N_CIRCLES)
    vars[0::3] = centers_x
    vars[1::3] = centers_y
    vars[2::3] = radii
    return vars

def init_random(seed):
    """Random initial guess with small radii to ensure feasibility"""
    rng = np.random.default_rng(seed)
    centers_x = rng.uniform(0.15, 0.85, N_CIRCLES)
    centers_y = rng.uniform(0.15, 0.85, N_CIRCLES)
    radii = np.full(N_CIRCLES, 0.01)
    
    vars = np.empty(3 * N_CIRCLES)
    vars[0::3] = centers_x
    vars[1::3] = centers_y
    vars[2::3] = radii
    return vars

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes circle packing to maximize sum of radii.
    Returns (centers, radii, sum_radii)
    """
    bounds = get_bounds()
    constraint_def = {'type': 'ineq', 'fun': constraint_fun}
    
    best_vars = None
    best_sum = -np.inf
    
    # Prepare multiple starting points
    starts = [init_grid()]
    for s in range(5):
        starts.append(init_random(seed=s * 137 + 42))
        
    for x0 in starts:
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraint_def,
                options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
            )
            
            # Evaluate feasibility and objective
            c_vals = constraint_fun(res.x)
            min_c = np.min(c_vals)
            current_sum = -res.fun
            
            # Accept if sufficiently feasible and better objective
            if min_c >= -1e-7 and current_sum > best_sum:
                best_sum = current_sum
                best_vars = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization fails unexpectedly
    if best_vars is None:
        best_vars = init_grid()
        
    # Post-processing: ensure strict validity within tolerance
    c_vals = constraint_fun(best_vars)
    if np.min(c_vals) < -1e-10:
        # Slightly shrink radii until valid
        scale = 0.9995
        while np.min(constraint_fun(best_vars)) < -1e-10:
            best_vars[2::3] *= scale
            scale *= 0.9995
            
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    total_sum = np.sum(radii)
    
    return centers, radii, float(total_sum)
