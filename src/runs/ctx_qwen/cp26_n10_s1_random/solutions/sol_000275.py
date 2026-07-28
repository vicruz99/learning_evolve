# sol_000275 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000262 (state 4217c70f) state=0e2694bf sum of radii=2.594550 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, l) for l in lims]
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A = np.zeros((m, n))
    A[np.arange(m), idx_i] = 1.0
    A[np.arange(m), idx_j] = 1.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b = dists[idx_i, idx_j]
    
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success and np.all(res.x >= -1e-9):
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def lp_obj_wrapper(x_flat, centers_full, idx):
    """Wrapper for coordinate descent: maximizes LP sum by moving one center."""
    old = centers_full[idx].copy()
    centers_full[idx] = x_flat
    _, s = solve_lp(centers_full)
    centers_full[idx] = old
    return -s

def obj_func(v, n):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def con_func(v, n):
    """Inequality constraints >= 0 for valid packing."""
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dr = r[idx_i] + r[idx_j]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N
    rng = np.random.default_rng(42)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    configs = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4],
        [6,6,5,5,4], [5,6,6,5,4], [7,5,5,5,4], [6,5,5,5,5],
        [5,6,5,5,5], [5,5,6,6,4], [4,5,6,5,6], [6,4,6,5,5]
    ]
    
    # Generate diverse hexagonal initial configurations
    for pat in patterns:
        if sum(pat) != n: continue
        pts = []
        r0 = 0.102
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        cfg = np.array(pts[:n])
        mn, mx = cfg.min(axis=0), cfg.max(axis=0)
        span = mx - mn
        if span[0] > 0: cfg[:, 0] = (cfg[:, 0] - mn[0]) / span[0] * 0.86 + 0.07
        if span[1] > 0: cfg[:, 1] = (cfg[:, 1] - mn[1]) / span[1] * 0.86 + 0.07
        configs.append(cfg)
        
        for _ in range(5):
            p = cfg + rng.uniform(-0.025, 0.025, cfg.shape)
            configs.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(12):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # Optimization pipeline
    for cfg in configs:
        r, s = solve_lp(cfg)
        if s <= best_sum: continue
        
        curr_c = np.clip(cfg, 1e-4, 1.0-1e-4).copy()
        curr_s = s
        
        # Phase 1: Coordinate Descent on centers
        for _ in range(5):
            improved = False
            for i in range(n):
                x0 = curr_c[i].copy()
                try:
                    res = minimize(lp_obj_wrapper, x0, args=(curr_c, i), method='Nelder-Mead',
                                   options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
                    if np.isfinite(res.fun) and res.fun < -curr_s - 1e-7:
                        curr_c[i] = res.x
                        curr_s = -res.fun
                        improved = True
                except Exception:
                    pass
            if not improved:
                break
                
        curr_c = np.clip(curr_c, 1e-4, 1.0-1e-4)
        r, s = solve_lp(curr_c)
        if s > best_sum:
            best_sum = s
            best_centers = curr_c.copy()
            best_radii = r.copy()
            
        # Phase 2: Joint SLSQP Polish
        if best_centers is not None:
            cx, cy = best_centers[:, 0], best_centers[:, 1]
            v0 = np.concatenate([cx, cy, best_radii * 0.995])
            bounds_v = [(0.0, 1.0)]*(2*n) + [(1e-7, 0.5)]*n
            
            try:
                res = minimize(obj_func, v0, args=(n,), method='SLSQP', bounds=bounds_v,
                               constraints={'type': 'ineq', 'fun': con_func, 'args': (n,)},
                               options={'maxiter': 3000, 'ftol': 1e-13})
                if np.isfinite(res.fun):
                    c_new = np.column_stack((res.x[:n], res.x[n:2*n]))
                    r_new = res.x[2*n:]
                    r_lp, s_lp = solve_lp(c_new)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_new.copy()
                        best_radii = r_lp.copy()
            except Exception:
                pass

    # Fallback
    if best_centers is None:
        grid = np.array([(i * 0.19 + 0.09, j * 0.19 + 0.09) for j in range(5) for i in range(5)] + [[0.5, 0.5]])
        best_centers = grid[:n]
        best_radii, best_sum = solve_lp(best_centers)

    # Phase 3: Strict Safety Scaling to guarantee numerical validity
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
