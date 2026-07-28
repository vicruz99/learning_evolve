# sol_000105 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000075 (state 5b5bfa68) state=3b50bd9e sum of radii=2.169899 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_radii_lp(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = [(0.0, None)] * n
    
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
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def objective(vars, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars[2 * n:])

def constraint_func(vars, n):
    """Computes inequality constraints >= 0 for valid packing."""
    cx = vars[:n]
    cy = vars[n:2 * n]
    r = vars[2 * n:]
    
    c = []
    # Boundary constraints
    c.extend(cx - r)
    c.extend(1.0 - cx - r)
    c.extend(cy - r)
    c.extend(1.0 - cy - r)
    
    # Pairwise non-overlap constraints: dist^2 >= (ri + rj)^2
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.extend((cx_m**2 + cy_m**2 - r_m**2)[mask])
    
    return np.array(c)

def get_hex_init(r0):
    """Generates an initial hexagonal grid of 26 circles."""
    pts = []
    y = r0
    row = 0
    dy = np.sqrt(3) * r0
    while len(pts) < 26 and y + r0 < 1.0:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 < 1.0 and len(pts) < 26:
            pts.append([x, y])
            x += 2 * r0
        y += dy
        row += 1
    while len(pts) < 26:
        pts.append([0.5, 0.5])
    return np.array(pts[:26])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.2)] * n
    rng = np.random.RandomState(42)
    
    # Generate diverse initial configurations
    starts = []
    starts.append(get_hex_init(0.10))
    starts.append(get_hex_init(0.11))
    starts.append(get_hex_init(0.09))
    
    for _ in range(25):
        base = get_hex_init(0.10).copy()
        base += rng.uniform(-0.035, 0.035, base.shape)
        starts.append(np.clip(base, 0.05, 0.95))
        
    for _ in range(10):
        starts.append(np.clip(rng.uniform(0.15, 0.85, (n, 2)), 0.05, 0.95))
        
    for cfg in starts:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2 * n] = cfg[:, 1]
        x0[2 * n:] = 0.095
        
        try:
            res = minimize(
                objective, x0, method='SLSQP', bounds=bounds,
                constraints={'type': 'ineq', 'fun': constraint_func, 'args': (n,)},
                options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False}
            )
            
            c_opt = np.column_stack((res.x[:n], res.x[n:2 * n]))
            
            # Post-process with LP to get optimal radii for these centers
            r_lp, s_lp = solve_radii_lp(c_opt)
            if r_lp is not None:
                r_lp *= 0.99999
                s_new = np.sum(r_lp)
                if s_new > best_sum:
                    best_sum = s_new
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Refinement loop: alternate center optimization and LP radius assignment
    if best_centers is not None:
        for _ in range(4):
            x0_ref = np.zeros(3 * n)
            x0_ref[:n] = best_centers[:, 0]
            x0_ref[n:2 * n] = best_centers[:, 1]
            x0_ref[2 * n:] = best_radii
            
            try:
                res_ref = minimize(
                    objective, x0_ref, method='SLSQP', bounds=bounds,
                    constraints={'type': 'ineq', 'fun': constraint_func, 'args': (n,)},
                    options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False}
                )
                c_ref = np.column_stack((res_ref.x[:n], res_ref.x[n:2 * n]))
                r_ref, s_ref = solve_radii_lp(c_ref)
                if r_ref is not None:
                    r_ref *= 0.99999
                    s_new = np.sum(r_ref)
                    if s_new > best_sum:
                        best_sum = s_new
                        best_centers = c_ref.copy()
                        best_radii = r_ref.copy()
            except Exception:
                break

    # Fallback if optimization fails
    if best_centers is None:
        c_fb = get_hex_init(0.095)
        r_fb, s_fb = solve_radii_lp(c_fb)
        if r_fb is not None:
            r_fb *= 0.99999
            best_centers = c_fb
            best_radii = r_fb
            best_sum = np.sum(r_fb)
            
    return best_centers, best_radii, float(best_sum)
