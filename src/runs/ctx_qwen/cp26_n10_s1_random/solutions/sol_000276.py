# sol_000276 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000262 (state 4217c70f) state=cc798eee sum of radii=2.620444 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[idx_i, idx_j]
    
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    bounds = [(0.0, max(l, 1e-9)) for l in lims]
    
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return res.x, -res.fun, res
    return None, 0.0, None

def lp_grad(centers, res):
    """Computes gradient of LP sum of radii w.r.t centers using dual variables."""
    n = centers.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    grad = np.zeros_like(centers)
    try:
        lams = getattr(getattr(res, 'ineqlin', None), 'marginals', None)
        if lams is None: return grad
        valid = lams > 1e-7
        if not np.any(valid): return grad
        k = np.where(valid)[0]
        i, j = idx_i[k], idx_j[k]
        lam = lams[k]
        diff = centers[i] - centers[j]
        dist = np.linalg.norm(diff, axis=1)
        dist_safe = np.maximum(dist, 1e-12)
        factors = (lam / dist_safe)[:, np.newaxis]
        np.add.at(grad, i, diff * factors)
        np.add.at(grad, j, -diff * factors)
    except Exception:
        pass
    return grad

def obj_lp(x):
    """Objective and gradient for L-BFGS-B: maximize LP sum of radii."""
    c = x.reshape(N, 2)
    r, s, res = solve_lp(c)
    if r is None:
        return 1e6, np.zeros_like(x)
    g = lp_grad(c, res)
    return -s, -g.flatten()

def obj_eq(v):
    """Objective for equal-radius optimization: minimize negative shared radius t."""
    return -v[-1]

def cons_eq(v):
    """Inequality constraints >= 0 for equal-radius packing."""
    t = v[-1]
    xs = v[:N]
    ys = v[N:2*N]
    c = np.concatenate([xs - t, 1.0 - xs - t, ys - t, 1.0 - ys - t])
    dx = xs[:, np.newaxis] - xs[np.newaxis, :]
    dy = ys[:, np.newaxis] - ys[np.newaxis, :]
    d2 = dx**2 + dy**2
    idx = np.triu_indices(N, k=1)
    c = np.concatenate([c, d2[idx] - 4.0*t**2])
    return c

def obj_joint(v):
    """Objective for joint SLSQP: maximize sum of radii."""
    return -np.sum(v[2*N:])

def cons_joint(v):
    """Inequality constraints >= 0 for SLSQP joint optimization."""
    x, y, r = v[:N], v[N:2*N], v[2*N:]
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dr = r[:, np.newaxis] + r[np.newaxis, :]
    idx = np.triu_indices(N, k=1)
    c = np.concatenate([c, dx[idx]**2 + dy[idx]**2 - dr[idx]**2])
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse hexagonal initial configurations
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,5,5], [6,6,5,5,4]]
    configs = []
    for pat in patterns:
        if sum(pat) != N: continue
        pts = []
        r0 = 0.101
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx % 2 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= N: break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
        base = np.array(pts[:N])
        configs.append(base)
        for _ in range(4):
            p = base.copy()
            p += rng.uniform(-0.025, 0.025, p.shape)
            p = np.clip(p, 0.03, 0.97)
            configs.append(p)
            
    for _ in range(6):
        configs.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # Phase 2: Optimize equal radius for each config
    best_t = 0.0
    best_c_eq = None
    bounds_eq = [(0.0, 1.0)]*(2*N) + [(0.09, 0.12)]
    
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), [0.10]])
        try:
            res = minimize(obj_eq, x0, method='SLSQP', bounds=bounds_eq,
                           constraints={'type': 'ineq', 'fun': cons_eq},
                           options={'maxiter': 4000, 'ftol': 1e-14})
            if np.isfinite(res.fun) and res.x[-1] > best_t:
                best_t = res.x[-1]
                best_c_eq = res.x[:2*N].reshape(N, 2).copy()
        except Exception:
            pass
            
    if best_c_eq is None:
        best_c_eq = configs[0]
        
    current_c = best_c_eq.copy()
    
    # Phase 3: LP gradient ascent on centers (L-BFGS-B)
    for _ in range(5):
        try:
            res = minimize(obj_lp, current_c.flatten(), method='L-BFGS-B', jac=True,
                           bounds=[(0.001, 0.999)]*(2*N),
                           options={'maxiter': 1200, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                current_c = res.x.reshape(N, 2)
        except Exception:
            pass
            
    # Phase 4: Random restarts around current best
    for _ in range(12):
        x_pert = current_c.flatten() + rng.uniform(-0.018, 0.018, 2*N)
        x_pert = np.clip(x_pert, 0.01, 0.99)
        try:
            res = minimize(obj_lp, x_pert, method='L-BFGS-B', jac=True,
                           bounds=[(0.001, 0.999)]*(2*N),
                           options={'maxiter': 900, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                current_c = res.x.reshape(N, 2)
        except Exception:
            pass
            
    r_best, s_best, _ = solve_lp(current_c)
    
    # Phase 5: Joint SLSQP polish
    v0 = np.concatenate([current_c.flatten(), r_best])
    bounds_j = [(0.0, 1.0)]*(2*N) + [(1e-6, 0.5)]*N
    try:
        res_j = minimize(obj_joint, v0, method='SLSQP', bounds=bounds_j,
                         constraints={'type': 'ineq', 'fun': cons_joint},
                         options={'maxiter': 6000, 'ftol': 1e-14})
        if np.isfinite(res_j.fun):
            cx, cy, r_j = res_j.x[:N], res_j.x[N:2*N], res_j.x[2*N:]
            current_c = np.column_stack((cx, cy))
            r_best, s_best, _ = solve_lp(current_c)
    except Exception:
        pass
        
    # Fallback if optimization fails completely
    if r_best is None:
        grid = np.array([(i * 0.18 + 0.1, j * 0.18 + 0.1) for j in range(5) for i in range(5)] + [[0.5, 0.5]])
        current_c = grid[:N]
        r_best, s_best, _ = solve_lp(current_c)
        
    # Final strict safety scaling to guarantee numerical validity
    scale = 1.0
    for i in range(N):
        x, y = current_c[i]
        r = r_best[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(current_c[i,0] - current_c[j,0], 
                         current_c[i,1] - current_c[j,1])
            rs = r_best[i] + r_best[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    r_final = r_best * scale * 0.9999995
    s_final = float(np.sum(r_final))
    
    return current_c, r_final, s_final
