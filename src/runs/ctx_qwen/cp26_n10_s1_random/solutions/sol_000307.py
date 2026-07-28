# sol_000307 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000286 (state b9c01463) state=4064f63e sum of radii=2.300100 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp(centers, n, A_ub, idx_i, idx_j, m_pairs):
    """Solves LP to maximize sum of radii for fixed centers."""
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    
    dx = centers[idx_i, 0] - centers[idx_j, 0]
    dy = centers[idx_i, 1] - centers[idx_j, 1]
    dists = np.hypot(dx, dy)
    dists = np.maximum(dists, 1e-9)
    
    b_ub = dists
    bounds = [(0.0, lim) for lim in lims]
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun, res
    except Exception:
        pass
    return None, 0.0, None

def get_lp_and_grad(centers, n, A_ub, idx_i, idx_j, m_pairs):
    """Returns (sum_radii, gradient_wrt_centers) using LP and dual variables."""
    r, s, res = solve_lp(centers, n, A_ub, idx_i, idx_j, m_pairs)
    if r is None:
        return s, np.zeros((n, 2))
        
    grad = np.zeros((n, 2))
    try:
        marg = np.asarray(res.ineqlin.marginals)
    except AttributeError:
        try:
            marg = np.asarray(res.marginals.ineqlin)
        except AttributeError:
            marg = None
            
    if marg is not None:
        for k in range(m_pairs):
            lam = marg[k]
            if lam > 1e-9:
                i, j = idx_i[k], idx_j[k]
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = np.hypot(dx, dy)
                if d < 1e-12: continue
                fx = lam * dx / d
                fy = lam * dy / d
                grad[i, 0] += fx
                grad[i, 1] += fy
                grad[j, 0] -= fx
                grad[j, 1] -= fy
    return s, grad

def obj_lp(c_flat, n, A_ub, idx_i, idx_j, m_pairs):
    """Objective for L-BFGS-B: negative sum of radii and its gradient."""
    c = c_flat.reshape(n, 2)
    s, g = get_lp_and_grad(c, n, A_ub, idx_i, idx_j, m_pairs)
    return -s, -g.flatten()

def joint_obj(v, n):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def joint_cons(v, n, idx_i, idx_j):
    """Inequality constraints >= 0 for valid packing in SLSQP."""
    cx = v[:n]
    cy = v[n:2*n]
    r = v[2*n:]
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    dx = cx[idx_i] - cx[idx_j]
    dy = cy[idx_i] - cy[idx_j]
    d2 = dx**2 + dy**2
    rs = r[idx_i] + r[idx_j]
    c = np.concatenate([c, d2 - rs**2])
    return c

def run_packing():
    n = 26
    idx_i, idx_j = np.triu_indices(n, k=1)
    m_pairs = len(idx_i)
    
    # Precompute pairwise constraint matrix
    A_ub = np.zeros((m_pairs, n))
    for k in range(m_pairs):
        A_ub[k, idx_i[k]] = 1.0
        A_ub[k, idx_j[k]] = 1.0
        
    rng = np.random.default_rng(42)
    best_s = -1.0
    best_c = None
    best_r = None
    
    configs = []
    
    # 1. Hexagonal lattice patterns (various row distributions)
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4], [6,6,5,5,4],
                [5,6,6,4,5], [7,6,6,7], [5,5,5,5,6], [6,5,5,6,4], [5,4,6,6,5]]
    for pat in patterns:
        if sum(pat) < n: continue
        pts = []
        y = 0.10
        for ri, cnt in enumerate(pat):
            shift = 0.10 if ri % 2 == 1 else 0.0
            x = 0.10 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 0.20
            y += 0.10 * np.sqrt(3)
        configs.append(np.array(pts[:n]))
        
    # 2. Boundary/Edge aligned patterns (exploit corners and edges)
    for r0 in [0.08, 0.09, 0.10]:
        pts = []
        x = r0
        while x + r0 <= 1.0 and len(pts) < n:
            pts.append([x, r0])
            x += 2.0 * r0
        x = r0
        while x + r0 <= 1.0 and len(pts) < n:
            pts.append([x, 1.0 - r0])
            x += 2.0 * r0
        y = r0 + 0.1
        while y + r0 <= 1.0 and len(pts) < n:
            pts.append([r0, y])
            y += 2.0 * r0
        y = r0 + 0.1
        while y + r0 <= 1.0 and len(pts) < n:
            pts.append([1.0 - r0, y])
            y += 2.0 * r0
        while len(pts) < n:
            pts.append([0.5 + rng.uniform(-0.25, 0.25), 0.5 + rng.uniform(-0.25, 0.25)])
        configs.append(np.array(pts[:n]))
        
    # 3. Random dense placements
    for _ in range(12):
        configs.append(rng.uniform(0.05, 0.95, (n, 2)))
        
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * n)
    
    # Optimization loop over all configurations
    for cfg in configs:
        c = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        r, s, _ = solve_lp(c, n, A_ub, idx_i, idx_j, m_pairs)
        if r is None: continue
        
        # Phase 1: Gradient ascent on centers using LP duals
        lr = 0.02
        vel = np.zeros_like(c)
        beta = 0.6
        for _ in range(250):
            s_val, grad = get_lp_and_grad(c, n, A_ub, idx_i, idx_j, m_pairs)
            g_norm = np.linalg.norm(grad)
            if g_norm < 1e-7: break
            g_dir = grad / g_norm
            vel = beta * vel + lr * g_dir
            c_new = c + vel
            c_new = np.clip(c_new, 1e-4, 1.0 - 1e-4)
            
            r_new, s_new, _ = solve_lp(c_new, n, A_ub, idx_i, idx_j, m_pairs)
            if r_new is not None and s_new > s_val + 1e-8:
                c = c_new
                s = s_new
                lr = min(lr * 1.05, 0.06)
            else:
                lr *= 0.85
                vel *= 0.5
            if lr < 1e-5: break
            
        # Phase 2: SLSQP Joint Polish for precise constraint handling
        r_cur, s_cur, _ = solve_lp(c, n, A_ub, idx_i, idx_j, m_pairs)
        if r_cur is not None:
            v0 = np.concatenate([c.flatten(), r_cur * 0.995])
            bounds_v = [(1e-4, 1.0-1e-4)]*(2*n) + [(1e-6, 0.5)]*n
            try:
                res = minimize(joint_obj, v0, args=(n,), method='SLSQP', bounds=bounds_v,
                               constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n, idx_i, idx_j)},
                               options={'maxiter': 4000, 'ftol': 1e-13})
                if np.isfinite(res.fun):
                    c_slq = res.x[:2*n].reshape(n, 2)
                    r_slq, s_slq, _ = solve_lp(c_slq, n, A_ub, idx_i, idx_j, m_pairs)
                    if r_slq is not None and s_slq > s:
                        c = c_slq
                        s = s_slq
            except Exception:
                pass
                
        if s > best_s:
            best_s = s
            best_c = c.copy()
            best_r, _, _ = solve_lp(c, n, A_ub, idx_i, idx_j, m_pairs)
            
    # Phase 3: Multi-scale perturbation search to escape local minima
    if best_c is not None:
        for scale in [0.005, 0.002, 0.0005]:
            for _ in range(25):
                c_pert = best_c + rng.uniform(-scale, scale, (n, 2))
                c_pert = np.clip(c_pert, 1e-4, 1.0 - 1e-4)
                try:
                    res_p = minimize(obj_lp, c_pert.flatten(), args=(n, A_ub, idx_i, idx_j, m_pairs),
                                     method='L-BFGS-B', jac=True, bounds=bounds_c,
                                     options={'maxiter': 1500, 'ftol': 1e-13})
                    if np.isfinite(res_p.fun):
                        c_p = res_p.x.reshape(n, 2)
                        r_p, s_p, _ = solve_lp(c_p, n, A_ub, idx_i, idx_j, m_pairs)
                        if r_p is not None and s_p > best_s:
                            best_s = s_p
                            best_c = c_p.copy()
                            best_r = r_p.copy()
                except Exception:
                    pass
                    
    # Final strict safety scaling to guarantee numerical validity
    if best_c is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_c[i,0], best_c[i,1], best_r[i]
            if r > 1e-12:
                scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
                rs = best_r[i] + best_r[j]
                if rs > 1e-12:
                    scale = min(scale, d/rs)
        best_r *= scale * 0.9999999
        best_s = float(np.sum(best_r))
    else:
        best_c = np.random.uniform(0.2, 0.8, (n, 2))
        best_r = np.full(n, 0.05)
        best_s = float(np.sum(best_r))
        
    return best_c, best_r, best_s
