# sol_000179 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000104 (state 0eb63b58) state=5be55752 sum of radii=2.624846 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = [(0.0, 0.5)] * n
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        lim = min(centers[i, 0], 1.0 - centers[i, 0], 
                  centers[i, 1], 1.0 - centers[i, 1])
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(lim)
        
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], 
                         centers[i, 1] - centers[j, 1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    try:
        res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), 
                      bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def objective(vars_arr, n):
    """Objective: minimize negative sum of radii => maximize sum of radii"""
    return -np.sum(vars_arr[2 * n:])

def constraint_func(vars_arr, n):
    """
    Computes inequality constraints >= 0 for valid packing.
    Constraints:
    1. Boundary: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    2. Non-overlap: (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    """
    cx = vars_arr[:n]
    cy = vars_arr[n:2 * n]
    r = vars_arr[2 * n:]
    
    c_list = []
    # Boundary constraints
    c_list.append(cx - r)
    c_list.append(1.0 - cx - r)
    c_list.append(cy - r)
    c_list.append(1.0 - cy - r)
    
    # Pairwise non-overlap constraints (vectorized)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = cx[idx_i] - cx[idx_j]
    dy = cy[idx_i] - cy[idx_j]
    rs = r[idx_i] + r[idx_j]
    c_list.append(dx**2 + dy**2 - rs**2)
    
    return np.concatenate(c_list)

def make_hex(rows_pattern, r0=0.07):
    """Generates initial positions on a hexagonal lattice."""
    n = 26
    pts = []
    y = r0
    for ri, cnt in enumerate(rows_pattern):
        shift = r0 if ri % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n:
                break
            pts.append([x, y])
            x += 2 * r0
        y += np.sqrt(3) * r0
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(123)
    inits = []
    
    # Structured hexagonal patterns summing to 26
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [5, 7, 5, 5, 4], [6, 6, 5, 5, 4],
        [5, 6, 4, 6, 5], [4, 5, 6, 5, 6], [5, 4, 6, 5, 6]
    ]
    
    for p in patterns:
        if sum(p) < n:
            continue
        pts = make_hex(p, 0.07)
        # Normalize and scale to fit comfortably inside [0.1, 0.9]
        pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0))
        pts = pts * 0.8 + 0.1
        inits.append(pts)
        # Add controlled perturbations to escape local minima
        for _ in range(2):
            inits.append(np.clip(pts + rng.uniform(-0.015, 0.015, pts.shape), 0.05, 0.95))
            
    # Random starts for robustness
    for _ in range(15):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))

    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-5, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraint_func, 'args': (n,)}

    # Phase 1: Broad search with NLP
    for cfg in inits:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2 * n] = cfg[:, 1]
        x0[2 * n:] = 0.07  # Feasible initial radius
        
        try:
            res = minimize(
                objective, x0, args=(n,),
                method='SLSQP', bounds=bounds,
                constraints=cons_dict, 
                options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False}
            )
            
            cx_opt = res.x[:n]
            cy_opt = res.x[n:2 * n]
            c_opt = np.column_stack((cx_opt, cy_opt))
            
            # Phase 1b: LP refinement for radii given optimized centers
            r_lp, s_lp = solve_lp_radii(c_opt)
            if r_lp is not None:
                r_lp *= 0.999999
                s_new = np.sum(r_lp)
                if s_new > best_sum:
                    best_sum = s_new
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue

    # Phase 2: Iterative joint refinement
    if best_centers is not None:
        for _ in range(6):
            x0_ref = np.zeros(3 * n)
            x0_ref[:n] = best_centers[:, 0]
            x0_ref[n:2 * n] = best_centers[:, 1]
            x0_ref[2 * n:] = best_radii
            
            try:
                res_ref = minimize(
                    objective, x0_ref, args=(n,),
                    method='SLSQP', bounds=bounds,
                    constraints=cons_dict, 
                    options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False}
                )
                c_ref = np.column_stack((res_ref.x[:n], res_ref.x[n:2 * n]))
                r_ref, s_ref = solve_lp_radii(c_ref)
                if r_ref is not None:
                    r_ref *= 0.999999
                    if s_ref > best_sum:
                        best_sum = s_ref
                        best_centers = c_ref.copy()
                        best_radii = r_ref.copy()
            except Exception:
                break

    # Fallback configuration if optimization fails unexpectedly
    if best_centers is None:
        pts = make_hex([5, 6, 5, 6, 4], 0.09)
        best_centers = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0))
        best_centers = best_centers * 0.8 + 0.1
        best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)

    # Final strict safety scaling to guarantee validity against 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
