# sol_000083 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000032 (state ac51bd1a) state=c18669fe sum of radii=2.329465 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def penalty_obj(vars_arr, n, mu):
    """Objective for equal-radius optimization: maximize t - penalty(violations)"""
    c = vars_arr[:2*n].reshape(n, 2)
    t = vars_arr[2*n]
    
    # Boundary penalties
    b_pen = np.sum(np.maximum(0, t - c[:, 0])**2)
    b_pen += np.sum(np.maximum(0, t - (1.0 - c[:, 0]))**2)
    b_pen += np.sum(np.maximum(0, t - c[:, 1])**2)
    b_pen += np.sum(np.maximum(0, t - (1.0 - c[:, 1]))**2)
    
    # Overlap penalties
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    o_pen = np.sum(np.maximum(0, 2*t - dists)**2)
    
    return -t + mu * (b_pen + o_pen)

def radii_obj(v, n):
    """Objective for radius refinement: maximize sum of radii"""
    return -np.sum(v)

def radii_cons(v, n, c_x, c_y):
    """Inequality constraints >= 0 for radius optimization"""
    r = v
    # Boundary constraints
    con = np.concatenate([c_x - r, 1.0 - c_x - r, c_y - r, 1.0 - c_y - r])
    # Pairwise non-overlap constraints
    dx = c_x[:, None] - c_x[None, :]
    dy = c_y[:, None] - c_y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[:, None] + r[None, :]
    triu = np.triu_indices(n, k=1)
    con = np.concatenate([con, dist[triu] - r_sum[triu]])
    return con

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    mu = 50000.0
    
    # 1. Generate initial hexagonal lattice
    pts = []
    r0 = 0.1
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
    base_c = np.array(pts[:n])
    
    best_t = 0.0
    best_c = None
    bounds_c = [(0.0, 1.0)] * (2*n) + [(0.01, 0.2)]
    
    np.random.seed(42)
    # 2. Optimize centers for equal radii with multiple restarts
    for trial in range(20):
        if trial == 0:
            c_init = base_c.copy()
        else:
            c_init = base_c + np.random.uniform(-0.04, 0.04, (n, 2))
            c_init = np.clip(c_init, 0.05, 0.95)
            
        x0 = np.concatenate([c_init.flatten(), [0.09]])
        
        try:
            res = minimize(penalty_obj, x0, args=(n, mu), method='L-BFGS-B', 
                           bounds=bounds_c, options={'maxiter': 15000, 'ftol': 1e-14})
            
            t_opt = res.x[-1]
            c_opt = res.x[:2*n].reshape(n, 2)
            
            # Compute exact feasible radius for this configuration
            min_wall = np.minimum(np.minimum(c_opt[:, 0], 1.0 - c_opt[:, 0]),
                                  np.minimum(c_opt[:, 1], 1.0 - c_opt[:, 1]))
            diff = c_opt[:, np.newaxis, :] - c_opt[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_pair = np.min(dists) / 2.0
            
            t_feas = min(min(min_wall), min_pair)
            
            if t_feas > best_t:
                best_t = t_feas
                best_c = c_opt.copy()
        except Exception:
            continue
            
    if best_c is None:
        best_c = base_c
        best_t = 0.08
        
    # 3. Refine radii independently given optimized centers
    c_x = best_c[:, 0]
    c_y = best_c[:, 1]
    r0_init = np.full(n, best_t * 0.95)
    bounds_r = [(0.0, 0.5)] * n
    
    try:
        res_r = minimize(radii_obj, r0_init, args=(n,), method='SLSQP', bounds=bounds_r,
                         constraints={'type': 'ineq', 'fun': radii_cons, 'args': (n, c_x, c_y)},
                         options={'maxiter': 3000, 'ftol': 1e-12})
        if np.isfinite(res_r.fun):
            best_r = res_r.x
        else:
            best_r = np.full(n, best_t)
    except Exception:
        best_r = np.full(n, best_t)
        
    # 4. Safety scaling to guarantee strict validity within 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y = best_c[i]
        r = best_r[i]
        if r < 1e-9: continue
        scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_c[i] - best_c[j])
            rs = best_r[i] + best_r[j]
            if rs < 1e-9: continue
            scale = min(scale, d / rs)
            
    best_r *= scale * 0.999999
    best_sum = np.sum(best_r)
    
    return best_c, best_r, float(best_sum)
