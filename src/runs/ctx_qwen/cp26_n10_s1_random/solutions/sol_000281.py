# sol_000281 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000265 (state 3cd04161) state=a744af5d sum of radii=2.602413 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_and_grad(centers, n, pairs, A_ub, m_con):
    b = np.zeros(m_con)
    for i in range(n):
        x, y = centers[i]
        b[i] = x
        b[n + i] = 1.0 - x
        b[2*n + i] = y
        b[3*n + i] = 1.0 - y
        
    for k, (i, j) in enumerate(pairs):
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        b[4*n + k] = np.hypot(dx, dy)
        
    bounds_r = [(0.0, None)] * n
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b, bounds=bounds_r, method='highs')
        if not res.success or not np.isfinite(res.fun):
            return None, None, None
            
        r = res.x
        s = -res.fun
        
        grad_c = np.zeros((n, 2))
        try:
            marg = res.ineqlin.marginals
        except AttributeError:
            try:
                marg = res.marginals.ineqlin
            except AttributeError:
                marg = None
                
        if marg is not None:
            for i in range(n):
                mu_x = marg[i]
                mu_1x = marg[n + i]
                mu_y = marg[2*n + i]
                mu_1y = marg[3*n + i]
                grad_c[i, 0] += mu_x - mu_1x
                grad_c[i, 1] += mu_y - mu_1y
                
            for k, (i, j) in enumerate(pairs):
                lam = marg[4*n + k]
                if lam > 1e-9:
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    d = b[4*n + k]
                    d_safe = max(d, 1e-9)
                    fx = lam * dx / d_safe
                    fy = lam * dy / d_safe
                    grad_c[i, 0] += fx
                    grad_c[i, 1] += fy
                    grad_c[j, 0] -= fx
                    grad_c[j, 1] -= fy
                    
        return r, s, grad_c
    except Exception:
        return None, None, None

def obj_lp_wrapper(c_flat, n, pairs, A_ub, m_con):
    c = c_flat.reshape(n, 2)
    _, s, _ = solve_lp_and_grad(c, n, pairs, A_ub, m_con)
    return -s if s is not None else 1e6

def grad_lp_wrapper(c_flat, n, pairs, A_ub, m_con):
    c = c_flat.reshape(n, 2)
    _, _, g = solve_lp_and_grad(c, n, pairs, A_ub, m_con)
    return -g.flatten() if g is not None else np.zeros_like(c_flat)

def solve_lp(centers, n):
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds_r = [(0.0, l) for l in lims]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1e9)
    
    m = n * (n - 1) // 2
    A = np.zeros((m, n))
    b = np.zeros(m)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[k, i] = 1.0
            A[k, j] = 1.0
            b[k] = dists[i, j]
            k += 1
            
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds_r, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def joint_obj(v, n):
    return -np.sum(v[2*n:])

def joint_cons(v, n):
    cx = v[:n]
    cy = v[n:2*n]
    r = v[2*n:]
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = cx[idx_i] - cx[idx_j]
    dy = cy[idx_i] - cy[idx_j]
    d2 = dx**2 + dy**2
    rs = r[idx_i] + r[idx_j]
    c = np.concatenate([c, d2 - rs**2])
    return c

def eq_obj(v, n):
    return -v[-1]

def eq_cons(v, n):
    xs = v[:n]
    ys = v[n:2*n]
    t = v[-1]
    c = np.concatenate([xs - t, 1.0 - xs - t, ys - t, 1.0 - ys - t])
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    c = np.concatenate([c, dx**2 + dy**2 - 4.0 * t**2])
    return c

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    
    # Generate hexagonal initial configurations
    configs = []
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [5, 7, 5, 5, 4],
        [6, 5, 5, 6, 4], [5, 6, 6, 4, 5], [7, 6, 6, 7], [8, 6, 6, 6]
    ]
    
    for pat in patterns:
        if sum(pat) < n: 
            continue
        r0 = 0.10
        pts = []
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: 
                    break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
        pts = np.array(pts[:n])
        configs.append(pts)
        
    for _ in range(6):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    best_t = 0.0
    best_centers_eq = None
    
    bounds_eq = [(0.0, 1.0)] * (2 * n) + [(0.08, 0.11)]
    
    # Phase 1: Equal radius optimization to find dense baseline
    for cfg in configs:
        for _ in range(3):
            c_pert = cfg + rng.uniform(-0.015, 0.015, (n, 2))
            c_pert = np.clip(c_pert, 0.05, 0.95)
            v0 = np.concatenate([c_pert.flatten(), [0.095]])
            try:
                res = minimize(eq_obj, v0, args=(n,), method='SLSQP', bounds=bounds_eq,
                               constraints={'type': 'ineq', 'fun': eq_cons, 'args': (n,)},
                               options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                if res.success and -res.fun > best_t:
                    best_t = -res.fun
                    best_centers_eq = res.x[:2*n].reshape(n, 2).copy()
            except Exception:
                pass
                
    if best_centers_eq is None:
        best_centers_eq = configs[0]
        
    # Phase 2: Joint SLSQP optimization for variable radii
    best_c = best_centers_eq.copy()
    best_r, best_s = solve_lp(best_c, n)
    
    v0 = np.concatenate([best_c[:, 0], best_c[:, 1], best_r * 0.99])
    bounds_slqp = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    for _ in range(5):
        v_start = v0 + rng.uniform(-0.002, 0.002, len(v0))
        v_start[:2*n] = np.clip(v_start[:2*n], 0.01, 0.99)
        try:
            res = minimize(joint_obj, v_start, args=(n,), method='SLSQP', bounds=bounds_slqp,
                           constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n,)},
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if np.isfinite(res.fun):
                c_new = np.column_stack((res.x[:n], res.x[n:2*n]))
                r_new, s_new = solve_lp(c_new, n)
                if s_new > best_s:
                    best_s = s_new
                    best_c = c_new.copy()
                    best_r = r_new.copy()
        except Exception:
            pass
            
    # Phase 3: LP dual gradient ascent for precise center refinement
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pairs = len(pairs)
    m_con = 4 * n + n_pairs
    A_ub = np.zeros((m_con, n))
    for i in range(n):
        A_ub[i, i] = 1.0
        A_ub[n + i, i] = 1.0
        A_ub[2*n + i, i] = 1.0
        A_ub[3*n + i, i] = 1.0
    for k, (i, j) in enumerate(pairs):
        A_ub[4*n + k, i] = 1.0
        A_ub[4*n + k, j] = 1.0
        
    bounds_c = [(0.005, 0.995)] * (2 * n)
    
    for _ in range(8):
        c_start = best_c + rng.uniform(-0.004, 0.004, (n, 2))
        c_start = np.clip(c_start, 0.02, 0.98)
        try:
            res = minimize(obj_lp_wrapper, c_start.flatten(), jac=grad_lp_wrapper, 
                           args=(n, pairs, A_ub, m_con), method='L-BFGS-B', 
                           bounds=bounds_c, options={'maxiter': 2000, 'ftol': 1e-12})
            if np.isfinite(res.fun):
                c_new = res.x.reshape(n, 2)
                r_new, s_new, _ = solve_lp_and_grad(c_new, n, pairs, A_ub, m_con)
                if s_new is not None and s_new > best_s:
                    best_s = s_new
                    best_c = c_new.copy()
                    best_r = r_new.copy()
        except Exception:
            pass
            
    # Final strict safety scaling to guarantee numerical validity
    scale = 1.0
    for i in range(n):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_c[i, 0] - best_c[j, 0], best_c[i, 1] - best_c[j, 1])
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_r *= scale * 0.9999999
    best_s = float(np.sum(best_r))
    
    return best_c, best_r, best_s
