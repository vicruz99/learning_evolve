# sol_000228 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000188 (state 061cb89c) state=835768ec sum of radii=2.616196 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

def lp_optimize_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    A = np.zeros((n * (n - 1) // 2, n))
    b = np.zeros(n * (n - 1) // 2)
    bounds = []
    
    for i in range(n):
        x, y = centers[i]
        ub = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(ub, 1e-9)))
        
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = d
            idx += 1
            
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-7), 0.0

def constr(vars_arr, n, triu_i, triu_j):
    """Inequality constraints for non-overlap. Boundary is handled by parameterization."""
    r = vars_arr[:n]
    u = vars_arr[n:2*n]
    v = vars_arr[2*n:3*n]
    
    x = r + (1.0 - 2.0 * r) * u
    y = r + (1.0 - 2.0 * r) * v
    
    dx = x[triu_i] - x[triu_j]
    dy = y[triu_i] - y[triu_j]
    dr = r[triu_i] + r[triu_j]
    
    return dx**2 + dy**2 - dr**2

def obj(vars_arr, n):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(vars_arr[:n])

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    triu_i, triu_j = np.triu_indices(n, k=1)
    bounds_vars = [(1e-5, 0.4)] * n + [(0.0, 1.0)] * n + [(0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': constr, 'args': (n, triu_i, triu_j)}
    
    # Diverse hexagonal row distributions summing to 26
    row_pats = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4],
        [5, 5, 5, 5, 6], [6, 5, 5, 6, 4], [5, 6, 4, 6, 5],
        [5, 5, 5, 6, 5], [4, 5, 6, 5, 6], [6, 6, 6, 4, 4]
    ]
    
    rng = np.random.default_rng(42)
    inits = []
    
    for pat in row_pats:
        if sum(pat) < n:
            continue
        pts = []
        y = 0.08
        dy = 0.173
        dx = 0.20
        for ri, cnt in enumerate(pat):
            shift = dx * 0.5 if ri % 2 == 1 else 0.0
            x_start = 0.5 - (cnt - 1) * dx / 2.0 + shift
            for _ in range(cnt):
                if len(pts) < n:
                    pts.append([x_start, y])
                x_start += dx
            y += dy
            if len(pts) >= n:
                break
                
        cfg = np.array(pts[:n])
        cfg += rng.uniform(-0.015, 0.015, cfg.shape)
        cfg = np.clip(cfg, 0.05, 0.95)
        
        r0 = np.full(n, 0.04)
        u0 = np.clip((cfg[:, 0] - r0) / (1.0 - 2.0 * r0), 0.0, 1.0)
        v0 = np.clip((cfg[:, 1] - r0) / (1.0 - 2.0 * r0), 0.0, 1.0)
        inits.append(np.concatenate([r0, u0, v0]))
        
    # Random starts
    for _ in range(5):
        c = rng.uniform(0.2, 0.8, (n, 2))
        r0 = np.full(n, 0.03)
        u0 = np.clip((c[:, 0] - r0) / (1.0 - 2.0 * r0), 0.0, 1.0)
        v0 = np.clip((c[:, 1] - r0) / (1.0 - 2.0 * r0), 0.0, 1.0)
        inits.append(np.concatenate([r0, u0, v0]))
        
    # Phase 1: SLSQP joint optimization
    for v0 in inits:
        try:
            res = minimize(obj, v0, args=(n,), method='SLSQP', bounds=bounds_vars,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if np.isfinite(res.fun):
                r_opt = res.x[:n]
                u_opt = res.x[n:2*n]
                v_opt = res.x[2*n:3*n]
                x = r_opt + (1.0 - 2.0 * r_opt) * u_opt
                y = r_opt + (1.0 - 2.0 * r_opt) * v_opt
                centers = np.column_stack((x, y))
                
                r_lp, s_lp = lp_optimize_radii(centers)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = centers.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Hill climbing on centers with LP evaluation
    if best_centers is not None:
        step = 0.004
        for _ in range(600):
            i = rng.integers(n)
            old_c = best_centers[i].copy()
            best_centers[i] += rng.uniform(-step, step, 2)
            best_centers[i] = np.clip(best_centers[i], 0.01, 0.99)
            
            r_try, s_try = lp_optimize_radii(best_centers)
            if s_try > best_sum + 1e-9:
                best_sum = s_try
                best_radii = r_try.copy()
            else:
                best_centers[i] = old_c
            step *= 0.999
            
        # Phase 3: Second SLSQP pass from improved configuration
        r0 = best_radii.copy()
        u0 = np.clip((best_centers[:, 0] - r0) / (1.0 - 2.0 * r0), 0.0, 1.0)
        v0 = np.clip((best_centers[:, 1] - r0) / (1.0 - 2.0 * r0), 0.0, 1.0)
        v0 = np.concatenate([r0, u0, v0])
        
        try:
            res = minimize(obj, v0, args=(n,), method='SLSQP', bounds=bounds_vars,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if np.isfinite(res.fun):
                r_opt = res.x[:n]
                u_opt = res.x[n:2*n]
                v_opt = res.x[2*n:3*n]
                x = r_opt + (1.0 - 2.0 * r_opt) * u_opt
                y = r_opt + (1.0 - 2.0 * r_opt) * v_opt
                centers = np.column_stack((x, y))
                
                r_lp, s_lp = lp_optimize_radii(centers)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = centers.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass
            
        # Phase 4: Final safety scaling
        scale = 1.0
        for i in range(n):
            x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
            if r > 1e-9:
                scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
                
        dx = best_centers[triu_i, 0] - best_centers[triu_j, 0]
        dy = best_centers[triu_i, 1] - best_centers[triu_j, 1]
        d = np.sqrt(dx**2 + dy**2)
        rs = best_radii[triu_i] + best_radii[triu_j]
        if np.any(rs > 1e-9):
            scale = min(scale, np.min(d / np.maximum(rs, 1e-12)))
            
        best_radii *= scale * 0.9999999
        best_sum = float(np.sum(best_radii))
        
    # Fallback
    if best_centers is None:
        pts = [[0.1 + i * 0.18, 0.1 + j * 0.18] for j in range(5) for i in range(5)]
        pts.append([0.55, 0.55])
        best_centers = np.array(pts[:26])
        best_radii, best_sum = lp_optimize_radii(best_centers)
        
    return best_centers, best_radii, best_sum
