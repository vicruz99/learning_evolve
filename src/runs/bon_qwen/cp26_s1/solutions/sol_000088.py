# sol_000088 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9821b492) state=d99d2dcb sum of radii=2.635983 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import math

def obj_func(p):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(p[2::3])

def con_func(p):
    """Constraint function: boundary and non-overlap constraints."""
    n = 26
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c_boundary = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Overlap constraints: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    ii, jj = np.triu_indices(n, k=1)
    dx = x[ii] - x[jj]
    dy = y[ii] - y[jj]
    dr = r[ii] + r[jj]
    c_overlap = dx**2 + dy**2 - dr**2
    
    return np.concatenate([c_boundary, c_overlap])

def run_packing():
    n = 26
    cons = NonlinearConstraint(con_func, 0, np.inf)
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    best_p = None
    best_val = -np.inf
    
    # Multi-start optimization
    for seed in range(10):
        np.random.seed(seed)
        
        # Hexagonal lattice initialization
        pts = []
        step = 0.22
        y = step * 0.5
        row = 0
        while y < 1.0 - step * 0.5:
            x = step * 0.5
            offset = (step * math.sqrt(3) / 2) if row % 2 else 0
            while x < 1.0 - step * 0.5:
                pts.append((x + offset, y))
                x += step * math.sqrt(3)
            y += step * 0.5 * math.sqrt(3)
            row += 1
            if len(pts) >= n:
                break
        
        # Fill remaining points randomly if needed
        while len(pts) < n:
            pts.append((np.random.rand() * 0.6 + 0.2, np.random.rand() * 0.6 + 0.2))
        pts = pts[:n]
        
        p0 = np.zeros(3 * n)
        p0[0::3] = [p[0] for p in pts]
        p0[1::3] = [p[1] for p in pts]
        p0[2::3] = np.full(n, 0.04)
        
        # Add small random perturbations to break symmetry
        p0[0::3] += np.random.uniform(-0.02, 0.02, n)
        p0[1::3] += np.random.uniform(-0.02, 0.02, n)
        p0[0::3] = np.clip(p0[0::3], 0.05, 0.95)
        p0[1::3] = np.clip(p0[1::3], 0.05, 0.95)
        
        try:
            res = minimize(obj_func, p0, method='SLSQP', bounds=bounds, constraints=[cons],
                           options={'ftol': 1e-10, 'maxiter': 2000, 'disp': False})
            val = -res.fun
            if val > best_val:
                best_val = val
                best_p = res.x.copy()
        except Exception:
            continue

    if best_p is None:
        best_p = p0.copy()
        best_val = -obj_func(p0)
        
    # Polishing phase to refine the best solution found
    try:
        res2 = minimize(obj_func, best_p, method='SLSQP', bounds=bounds, constraints=[cons],
                        options={'ftol': 1e-12, 'maxiter': 3000, 'disp': False})
        if -res2.fun > best_val:
            best_val = -res2.fun
            best_p = res2.x.copy()
    except Exception:
        pass

    centers = np.column_stack((best_p[0::3], best_p[1::3]))
    radii = best_p[2::3]
    
    return centers, radii, best_val
