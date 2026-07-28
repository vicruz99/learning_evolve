# sol_000178 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000104 (state 0eb63b58) state=a8b82f3e sum of radii=2.635980 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """
    Given fixed centers, solves the LP to maximize sum of radii.
    Constraints: r_i + r_j <= dist(i, j) and r_i <= distance to boundary.
    """
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = [(0.0, 0.5)] * n
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        lim = min(x, 1.0 - x, y, 1.0 - y)
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(lim)
        
    # Pairwise non-overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
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

def objective_joint(vars_array, n):
    """Objective: minimize negative sum of radii"""
    return -np.sum(vars_array[2 * n:])

def constraint_func_joint(vars_array, n):
    """
    Computes inequality constraints >= 0 for valid packing.
    Uses squared distance to avoid square roots during optimization.
    """
    cx = vars_array[:n]
    cy = vars_array[n:2 * n]
    r = vars_array[2 * n:]
    
    # Boundary constraints
    b = np.empty(4 * n)
    b[:n] = cx - r
    b[n:2 * n] = 1.0 - cx - r
    b[2 * n:3 * n] = cy - r
    b[3 * n:4 * n] = 1.0 - cy - r
    
    # Pairwise non-overlap constraints: dist^2 >= (ri + rj)^2
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    
    d2 = cx_m**2 + cy_m**2
    rs2 = r_m**2
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    p = d2[idx_i, idx_j] - rs2[idx_i, idx_j]
    
    return np.concatenate([b, p])

def apply_safety_scale(centers, radii, n):
    """Computes and applies a safety scale to guarantee strict validity."""
    scale = 1.0
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            rs = radii[i] + radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    return radii * scale * 0.999999

def make_hex(rows_pattern, r0=0.09):
    """Generates initial positions on a hexagonal lattice with specified row distribution."""
    n = 26
    pts = []
    y = r0
    for ri, cnt in enumerate(rows_pattern):
        shift = r0 if ri % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n: break
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
    
    rng = np.random.default_rng(42)
    
    # Diverse initial configurations
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5],
        [6, 5, 5, 6, 4], [5, 5, 4, 6, 6], [4, 5, 6, 5, 6], [5, 4, 6, 5, 6]
    ]
    for p in patterns:
        if sum(p) < n: continue
        pts = make_hex(p)
        inits.append(pts)
        for _ in range(6):
            pts_p = pts + rng.uniform(-0.025, 0.025, pts.shape)
            inits.append(np.clip(pts_p, 0.05, 0.95))
            
    for _ in range(12):
        inits.append(rng.uniform(0.12, 0.88, (n, 2)))
        
    g = np.linspace(0.15, 0.85, 5)
    grid = np.array([[x, y] for y in g for x in g])
    grid = np.vstack([grid, [0.5, 0.5]])
    inits.append(grid)
    
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraint_func_joint, 'args': (n,)}
    
    # Phase 1: Joint SLSQP optimization + LP refinement
    for cfg in inits:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2 * n] = cfg[:, 1]
        x0[2 * n:] = 0.085
        
        try:
            res = minimize(objective_joint, x0, args=(n,),
                           method='SLSQP', bounds=bounds_vars,
                           constraints=cons_dict, 
                           options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            
            cx = res.x[:n]
            cy = res.x[n:2 * n]
            centers = np.column_stack((cx, cy))
            
            # LP refinement for exact radius maximization
            r_lp, s_lp = solve_lp_radii(centers)
            if r_lp is not None:
                r_final = apply_safety_scale(centers, r_lp, n)
                s_final = np.sum(r_final)
                
                if s_final > best_sum:
                    best_sum = s_final
                    best_centers = centers.copy()
                    best_radii = r_final.copy()
        except Exception:
            continue

    # Phase 2: Local search on centers to maximize LP sum
    if best_centers is not None:
        def lp_obj(centers_flat):
            c = centers_flat.reshape(n, 2)
            _, s = solve_lp_radii(c)
            return -s

        for _ in range(4):
            try:
                c_start = best_centers + rng.uniform(-0.006, 0.006, best_centers.shape)
                c_start = np.clip(c_start, 0.05, 0.95)
                
                res_loc = minimize(lp_obj, c_start.flatten(), method='Nelder-Mead',
                                   options={'maxiter': 400, 'xatol': 1e-6, 'fatol': 1e-8})
                c_opt = res_loc.x.reshape(n, 2)
                
                r_lp, s_lp = solve_lp_radii(c_opt)
                if r_lp is not None:
                    r_final = apply_safety_scale(c_opt, r_lp, n)
                    s_final = np.sum(r_final)
                    
                    if s_final > best_sum:
                        best_sum = s_final
                        best_centers = c_opt.copy()
                        best_radii = r_final.copy()
            except Exception:
                continue

    # Fallback configuration
    if best_centers is None:
        best_centers = make_hex([5, 6, 5, 6, 4])
        r_fb, _ = solve_lp_radii(best_centers)
        if r_fb is not None:
            best_radii = apply_safety_scale(best_centers, r_fb, n)
        else:
            best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum)
