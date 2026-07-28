# sol_000341 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000152 (state 06e8663d) state=774c5f59 sum of radii=2.625404 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def objective(vars_arr, n):
    """Minimize negative sum of radii => Maximize sum of radii"""
    return -np.sum(vars_arr[2*n:])

def constraint_func(vars_arr, n):
    """Computes inequality constraints >= 0 for valid packing"""
    cx, cy, r = vars_arr[:n], vars_arr[n:2*n], vars_arr[2*n:]
    c = []
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c.append(cx - r)
    c.append(1.0 - cx - r)
    c.append(cy - r)
    c.append(1.0 - cy - r)
    
    # Pairwise non-overlap constraints: dist(i,j) >= r_i + r_j
    # Using direct Euclidean distance provides better gradient behavior than squared distance
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.hypot(dx, dy)
    r_sum = r[:, None] + r[None, :]
    
    ii, jj = np.triu_indices(n, k=1)
    c.append(dist[ii, jj] - r_sum[ii, jj])
    
    return np.concatenate(c)

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((1e-9, max(mx, 1e-9)))
        
    A_ub = np.zeros((n*(n-1)//2, n))
    b_ub = np.zeros(n*(n-1)//2)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def generate_hex_init(row_counts, n, scale=0.85):
    """Generates a hexagonal lattice configuration for N circles"""
    r0 = 0.10
    dy = np.sqrt(3) * r0
    y = r0
    pts = []
    for idx, cnt in enumerate(row_counts):
        shift = r0 if idx % 2 == 1 else 0.0
        row_width = (cnt - 1) * 2 * r0
        x_start = (1.0 - row_width) / 2.0 + shift
        for _ in range(cnt):
            if len(pts) >= n: break
            pts.append([x_start, y])
            x_start += 2 * r0
        y += dy
        
    pts = np.array(pts[:n])
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn
    if span[0] < 1e-8: span[0] = 1.0
    if span[1] < 1e-8: span[1] = 1.0
    pts = (pts - mn) / span * scale + (1.0 - scale) / 2.0
    return np.clip(pts, 0.02, 0.98)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(12345)
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-5, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func, 'args': (n,)}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Diverse row distributions known to yield high packing densities for N~26
    row_configs = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 6, 4],
        [4, 6, 6, 5, 5], [6, 6, 5, 5, 4], [5, 6, 5, 5, 5],
        [7, 5, 6, 4], [5, 7, 6, 4], [4, 7, 6, 5], [6, 4, 6, 6],
        [5, 6, 4, 6, 5], [6, 5, 4, 6, 5], [5, 5, 5, 6, 5],
        [6, 6, 4, 5, 5], [4, 5, 6, 5, 6], [5, 4, 6, 6, 5]
    ]
    
    inits = []
    for rc in row_configs:
        if sum(rc) >= n:
            pts = generate_hex_init(rc, n, 0.88)
            inits.append(pts)
            for _ in range(4):
                p = pts + rng.uniform(-0.03, 0.03, pts.shape)
                inits.append(np.clip(p, 0.05, 0.95))
                
    for _ in range(10):
        inits.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # Phase 1: Multi-start SLSQP Joint Optimization
    for init_cfg in inits:
        x0 = np.zeros(3 * n)
        x0[:n] = init_cfg[:, 0]
        x0[n:2*n] = init_cfg[:, 1]
        x0[2*n:] = 0.065  # Start strictly feasible to guide SLSQP
        
        try:
            res = minimize(objective, x0, method='SLSQP', args=(n,),
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 10000, 'ftol': 1e-14})
            
            if np.isfinite(res.fun) and -res.fun > best_sum:
                c_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
                c_opt = np.clip(c_opt, 1e-6, 1.0 - 1e-6)
                lp_r, lp_s = solve_lp_radii(c_opt)
                if lp_r is not None and lp_s > best_sum:
                    best_sum = lp_s
                    best_centers = c_opt.copy()
                    best_radii = lp_r.copy()
        except Exception:
            continue
            
    # Phase 2: Iterative Jitter & LP/SLSQP Polish Refinement
    if best_centers is not None:
        curr_c, curr_r, curr_s = best_centers, best_radii, best_sum
        global_best_c, global_best_r, global_best_s = curr_c, curr_r, curr_s
        
        for epoch in range(35):
            for _ in range(8):
                noise_std = 0.006 * (0.92 ** epoch)
                pert_c = np.clip(curr_c + rng.normal(0, noise_std, (n, 2)), 0.01, 0.99)
                lp_r, lp_s = solve_lp_radii(pert_c)
                if lp_r is not None:
                    if lp_s > curr_s + 1e-9:
                        curr_s = lp_s
                        curr_c = pert_c.copy()
                        curr_r = lp_r.copy()
                    if lp_s > global_best_s + 1e-9:
                        global_best_s = lp_s
                        global_best_c = pert_c.copy()
                        global_best_r = lp_r.copy()
                        
            # Restart local search from global best if current trajectory stagnates
            if curr_s < global_best_s - 1e-5:
                curr_c, curr_r, curr_s = global_best_c, global_best_r, global_best_s
                
            # Polish best configuration with SLSQP
            x0 = np.zeros(3 * n)
            x0[:n] = curr_c[:, 0]
            x0[n:2*n] = curr_c[:, 1]
            x0[2*n:] = curr_r * 0.97
            try:
                res = minimize(objective, x0, method='SLSQP', args=(n,),
                               bounds=bounds, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-14})
                if np.isfinite(res.fun):
                    c_pol = np.column_stack((res.x[:n], res.x[n:2*n]))
                    lp_r, lp_s = solve_lp_radii(c_pol)
                    if lp_r is not None and lp_s > global_best_s:
                        global_best_s = lp_s
                        global_best_c = c_pol.copy()
                        global_best_r = lp_r.copy()
            except Exception:
                pass
                
        best_centers, best_radii, best_sum = global_best_c, global_best_r, global_best_s
        
    # Fallback safety configuration
    if best_centers is None:
        r_f = 0.08
        best_centers = np.array([(r_f + 0.18*i, r_f + 0.18*j) for j in range(5) for i in range(5)] + [[0.5, 0.5]])
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # Final strict safety scaling to guarantee numerical validity against 1e-12 tolerance
    if best_radii is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
            if r > 1e-9:
                scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
                rs = best_radii[i] + best_radii[j]
                if rs > 1e-9:
                    scale = min(scale, d/rs)
        best_radii *= scale * 0.9999999
        best_sum = float(np.sum(best_radii))
        
    return best_centers, best_radii, best_sum
