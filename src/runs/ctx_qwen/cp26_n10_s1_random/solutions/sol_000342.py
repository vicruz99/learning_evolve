# sol_000342 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000152 (state 06e8663d) state=13938c86 sum of radii=2.626566 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_coords(vars_arr, n):
    """Decode parameterized variables to (x, y) centers."""
    r = vars_arr[:n]
    u = vars_arr[n:2*n]
    v = vars_arr[2*n:3*n]
    x = r + (1.0 - 2.0 * r) * u
    y = r + (1.0 - 2.0 * r) * v
    return np.column_stack((x, y))

def obj(vars_arr, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_arr[:n])

def constraints(vars_arr, n):
    """Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2."""
    r = vars_arr[:n]
    u = vars_arr[n:2*n]
    v = vars_arr[2*n:3*n]
    x = r + (1.0 - 2.0 * r) * u
    y = r + (1.0 - 2.0 * r) * v
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    rs = (r[:, None] + r[None, :])**2
    
    idx = np.triu_indices(n, k=1)
    return d2[idx] - rs[idx]

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(mx, 1e-9)))
        
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
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
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def generate_hex_patterns(n, rng):
    """Generates diverse hexagonal lattice initial configurations."""
    patterns = []
    rows_cfgs = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [5, 6, 6, 4, 5],
        [7, 6, 6, 7], [6, 7, 6, 7], [5, 5, 5, 5, 6], [6, 5, 5, 5, 5],
        [5, 6, 5, 5, 5], [6, 5, 5, 6, 4]
    ]
    
    for rc in rows_cfgs:
        if sum(rc) < n: 
            continue
        r0 = 0.09
        y = r0
        pts = []
        for ri, cnt in enumerate(rc):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: 
                    break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
            
        pts = np.array(pts[:n])
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = mx - mn
        if span[0] > 1e-9: pts[:, 0] = (pts[:, 0] - mn[0]) / span[0] * 0.88 + 0.06
        if span[1] > 1e-9: pts[:, 1] = (pts[:, 1] - mn[1]) / span[1] * 0.88 + 0.06
        
        patterns.append(pts)
        for _ in range(4):
            p = pts + rng.uniform(-0.025, 0.025, pts.shape)
            patterns.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(6):
        patterns.append(rng.uniform(0.12, 0.88, (n, 2)))
        
    return patterns

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_opt = [(1e-6, 0.45)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n
    cons_dict = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    inits = generate_hex_patterns(n, rng)
    
    # Phase 1: SLSQP from diverse starts
    for cfg in inits:
        r0 = 0.09
        denom = 1.0 - 2.0 * r0
        u0 = np.clip((cfg[:, 0] - r0) / denom, 0.0, 1.0)
        v0 = np.clip((cfg[:, 1] - r0) / denom, 0.0, 1.0)
        x0 = np.concatenate([np.full(n, r0), u0, v0])
        
        try:
            res = minimize(obj, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                           constraints=cons_dict, options={'maxiter': 8000, 'ftol': 1e-13})
            
            if np.isfinite(res.fun):
                c_opt = get_coords(res.x, n)
                c_opt = np.clip(c_opt, 1e-6, 1.0 - 1e-6)
                lp_r, _ = solve_lp_radii(c_opt)
                if lp_r is not None:
                    s = np.sum(lp_r)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_opt.copy()
                        best_radii = lp_r.copy()
        except Exception:
            pass
            
    # Phase 2: Jiggle search to escape local minima
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        step = 0.006
        
        for it in range(3000):
            idx = rng.integers(n)
            old_c = curr_c[idx].copy()
            
            new_c = curr_c[idx] + rng.uniform(-step, step, 2)
            new_c = np.clip(new_c, 0.02, 0.98)
            curr_c[idx] = new_c
            
            lp_r, _ = solve_lp_radii(curr_c)
            if lp_r is not None:
                s = np.sum(lp_r)
                if s > curr_s + 1e-8:
                    curr_s = s
                    curr_r = lp_r.copy()
                    step *= 0.999
                    if s > best_sum:
                        best_sum = s
                        best_centers = curr_c.copy()
                        best_radii = curr_r.copy()
                else:
                    curr_c[idx] = old_c
                    if rng.random() < 0.01:
                        step *= 0.9
            else:
                curr_c[idx] = old_c
                
        # Phase 3: Polish with SLSQP on jiggle-refined configuration
        c_polish = best_centers.copy()
        r_init = best_radii * 0.99
        denom = 1.0 - 2.0 * r_init
        u_polish = np.clip((c_polish[:, 0] - r_init) / denom, 0.0, 1.0)
        v_polish = np.clip((c_polish[:, 1] - r_init) / denom, 0.0, 1.0)
        x0_polish = np.concatenate([r_init, u_polish, v_polish])
        
        try:
            res_polish = minimize(obj, x0_polish, args=(n,), method='SLSQP', bounds=bounds_opt,
                                  constraints=cons_dict, options={'maxiter': 6000, 'ftol': 1e-13})
            if np.isfinite(res_polish.fun):
                c_opt = get_coords(res_polish.x, n)
                c_opt = np.clip(c_opt, 1e-6, 1.0 - 1e-6)
                lp_r, _ = solve_lp_radii(c_opt)
                if lp_r is not None:
                    s = np.sum(lp_r)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_opt.copy()
                        best_radii = lp_r.copy()
        except Exception:
            pass

    # Fallback safety net
    if best_centers is None:
        best_centers = generate_hex_patterns(n, rng)[0]
        best_radii, _ = solve_lp_radii(best_centers)
        best_sum = np.sum(best_radii)
        
    # Final strict safety scaling to guarantee 1e-12 tolerance compliance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
