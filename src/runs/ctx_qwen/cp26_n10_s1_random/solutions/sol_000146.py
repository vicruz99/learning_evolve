# sol_000146 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000123 (state 90e3970d) state=3cc8e8a2 sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_constraints(vars_array, n):
    """Computes inequality constraints >= 0 for valid packing."""
    cx = vars_array[:n]
    cy = vars_array[n:2*n]
    r = vars_array[2*n:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    b1 = cx - r
    b2 = 1.0 - cx - r
    b3 = cy - r
    b4 = 1.0 - cy - r
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    cx_diff = cx[:, np.newaxis] - cx[np.newaxis, :]
    cy_diff = cy[:, np.newaxis] - cy[np.newaxis, :]
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    d2 = cx_diff**2 + cy_diff**2
    rs2 = r_sum**2
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    p_cons = d2[mask] - rs2[mask]
    
    return np.concatenate([b1, b2, b3, b4, p_cons])

def objective_func(vars_array, n):
    """Minimize negative sum of radii."""
    return -np.sum(vars_array[2*n:])

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c = -np.ones(n)
    bounds = [(0.0, None)] * n
    
    A_ub_list = []
    b_ub_list = []
    
    # Boundary constraints for radii
    for i in range(n):
        x, y = centers[i]
        lims = (x, 1.0 - x, y, 1.0 - y)
        for lim in lims:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(lim)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(d)
            
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def generate_hex_centers(n, r0):
    """Generates a hexagonal lattice initialization."""
    pts = []
    y = r0
    row = 0
    while len(pts) < n:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Generate diverse initial configurations
    inits = []
    for r0 in [0.09, 0.095, 0.10, 0.105]:
        centers = generate_hex_centers(n, r0)
        centers_p = centers + rng.uniform(-0.02, 0.02, centers.shape)
        centers_p = np.clip(centers_p, 0.05, 0.95)
        vars_init = np.zeros(3 * n)
        vars_init[:n] = centers_p[:, 0]
        vars_init[n:2*n] = centers_p[:, 1]
        vars_init[2*n:] = r0
        inits.append(vars_init)
        
    # Grid initialization
    gx = np.linspace(0.1, 0.9, 5)
    gy = np.linspace(0.1, 0.9, 5)
    grid_pts = np.array([(x, y) for y in gy for x in gx])
    grid_pts = np.vstack([grid_pts, [0.5, 0.5]])
    vars_grid = np.zeros(3 * n)
    vars_grid[:n] = grid_pts[:, 0]
    vars_grid[n:2*n] = grid_pts[:, 1]
    vars_grid[2*n:] = 0.09
    inits.append(vars_grid)
    
    current_best_vars = inits[0]
    
    # Optimization loop with hybrid SLSQP + LP refinement
    for trial in range(25):
        if trial == 0:
            x0 = inits[0].copy()
        elif trial < 10:
            x0 = inits[trial % len(inits)].copy()
            x0 += rng.uniform(-0.01, 0.01, x0.shape)
            x0 = np.clip(x0, 1e-6, 0.99)
            x0[2*n:] = np.clip(x0[2*n:], 1e-6, 0.5)
        else:
            # Perturb the best configuration found so far
            x0 = current_best_vars.copy()
            x0[:2*n] += rng.uniform(-0.005, 0.005, 2*n)
            x0[:2*n] = np.clip(x0[:2*n], 0.0, 1.0)
            
        try:
            res = minimize(objective_func, x0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            
            if res.success:
                c_val = compute_constraints(res.x, n)
                if np.min(c_val) > -1e-5:
                    cx = res.x[:n]
                    cy = res.x[n:2*n]
                    c_opt = np.column_stack((cx, cy))
                    
                    # LP refinement to extract maximal radii for these centers
                    lp_r, lp_sum = solve_lp_radii(c_opt)
                    if lp_r is not None:
                        # Tiny shrink to guarantee strict numerical validity against checker tolerance
                        lp_r *= 0.9999999
                        lp_sum = np.sum(lp_r)
                        if lp_sum > best_sum:
                            best_sum = lp_sum
                            best_centers = c_opt.copy()
                            best_radii = lp_r.copy()
                            current_best_vars = np.concatenate([c_opt[:, 0], c_opt[:, 1], lp_r])
        except Exception:
            pass
            
    # Fallback configuration
    if best_centers is None:
        best_centers = generate_hex_centers(n, 0.09)
        best_radii = np.full(n, 0.085)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)
