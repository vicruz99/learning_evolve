# sol_000059 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000035 (state cfcb3616) state=3e3cfdc0 sum of radii=2.629092 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective: minimize negative sum of radii (equivalent to maximizing sum)."""
    return -np.sum(vars[2::3])

def constraints(vars):
    """
    Inequality constraints: g(vars) >= 0.
    Enforces boundary containment and pairwise non-overlap.
    """
    n = len(vars) // 3
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    c = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist_sq = dx * dx + dy * dy
    r_sum = r[i_idx] + r[j_idx]
    c.append(dist_sq - r_sum * r_sum)
    
    return np.concatenate(c)

def make_init(n, seed):
    """Generates a strictly feasible initial configuration from structured layouts."""
    rng = np.random.RandomState(seed)
    mode = seed % 3
    pts = np.zeros((n, 2))
    
    if mode == 0:  # Hexagonal lattice
        r_est = 0.095
        y = r_est
        row = 0
        idx = 0
        while y <= 1.0 - r_est and idx < n:
            x = r_est + (row % 2) * r_est
            while x <= 1.0 - r_est and idx < n:
                pts[idx] = [x, y]
                idx += 1
                x += 2.0 * r_est
            y += np.sqrt(3.0) * r_est
            row += 1
    elif mode == 1:  # Square grid
        cols = 5
        rows = 6
        step_x = (1.0 - 2.0 * 0.05) / (cols - 1)
        step_y = (1.0 - 2.0 * 0.05) / (rows - 1)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx < n:
                    pts[idx] = [0.05 + j * step_x, 0.05 + i * step_y]
                    idx += 1
    else:  # Random spread
        pts[:, 0] = np.sort(rng.rand(n))
        pts[:, 1] = rng.rand(n)
        
    # Perturb to escape exact symmetries
    pts += rng.uniform(-0.04, 0.04, pts.shape)
    pts = np.clip(pts, 0.02, 0.98)
    
    # Compute strictly feasible initial radii
    r = np.zeros(n)
    for i in range(n):
        dw = min(pts[i, 0], 1.0 - pts[i, 0], pts[i, 1], 1.0 - pts[i, 1])
        dm = 1.0
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((pts[i] - pts[j]) ** 2))
                if d < dm:
                    dm = d
        # 0.85 factor ensures strict feasibility for SLSQP start
        r[i] = 0.85 * min(dw, dm / 2.0)
        
    vars0 = np.zeros(3 * n)
    vars0[0::3] = pts[:, 0]
    vars0[1::3] = pts[:, 1]
    vars0[2::3] = r
    return vars0

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_vars = None
    best_sum = -np.inf
    
    # Phase 1: Multiple restarts from diverse feasible configurations
    for seed in range(50):
        x0 = make_init(n, seed)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: High-precision refinement on the best configuration
    if best_vars is not None:
        try:
            res_final = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if res_final.success:
                c_val = constraints(res_final.x)
                if np.min(c_val) >= -1e-7:
                    best_vars = res_final.x
                    best_sum = -res_final.fun
        except Exception:
            pass
    else:
        # Fallback if optimization fails completely
        best_vars = make_init(n, 0)
        best_sum = np.sum(best_vars[2::3])
        
    r_opt = best_vars[2::3]
    x_opt = best_vars[0::3]
    y_opt = best_vars[1::3]
    centers = np.column_stack((x_opt, y_opt))
    return centers, r_opt, float(np.sum(r_opt))
