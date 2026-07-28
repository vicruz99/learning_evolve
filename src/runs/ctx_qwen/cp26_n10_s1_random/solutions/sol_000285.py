# sol_000285 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000265 (state 3cd04161) state=aceece95 sum of radii=2.621283 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_and_grad(centers, n, pairs, A_ub, m_con):
    """Solves LP for fixed centers and computes gradient of sum of radii w.r.t centers."""
    b = np.zeros(m_con)
    for i in range(n):
        b[i] = centers[i, 0]
        b[n + i] = 1.0 - centers[i, 0]
        b[2*n + i] = centers[i, 1]
        b[3*n + i] = 1.0 - centers[i, 1]
        
    for k, (i, j) in enumerate(pairs):
        dx = centers[i, 0] - centers[j, 0]
        dy = centers[i, 1] - centers[j, 1]
        b[4*n + k] = np.hypot(dx, dy)
        
    bounds_r = [(0.0, 1.0)] * n
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b, bounds=bounds_r, method='highs')
        if not res.success or not np.isfinite(res.fun):
            return None, None, None
            
        r = res.x
        s = -res.fun
        
        grad_c = np.zeros((n, 2))
        marg = None
        if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
            marg = res.marginals.ineqlin
        elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
            marg = res.ineqlin.marginals
            
        if marg is not None:
            # Boundary duals contribution
            for i in range(n):
                grad_c[i, 0] += marg[i] - marg[n + i]
                grad_c[i, 1] += marg[2*n + i] - marg[3*n + i]
                
            # Pairwise duals contribution
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

def obj_func(c_flat, n, pairs, A_ub, m_con):
    """Objective function for L-BFGS-B: negative sum of radii and its gradient."""
    c = c_flat.reshape(n, 2)
    r, s, g = solve_lp_and_grad(c, n, pairs, A_ub, m_con)
    if r is None:
        return 1e6, np.zeros_like(c_flat)
    return -s, -g.flatten()

def joint_obj(v, n):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def joint_cons(v, n):
    """Inequality constraints >= 0 for valid packing in SLSQP."""
    cx = v[:n]
    cy = v[n:2*n]
    r = v[2*n:]
    cons = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    triu_idx = np.triu_indices(n, k=1)
    dx = cx[triu_idx[0]] - cx[triu_idx[1]]
    dy = cy[triu_idx[0]] - cy[triu_idx[1]]
    d2 = dx**2 + dy**2
    rs = r[triu_idx[0]] + r[triu_idx[1]]
    cons = np.concatenate([cons, d2 - rs**2])
    return cons

def generate_hex_init(row_counts, r0, n):
    """Generates a hexagonal lattice initialization with specified row counts."""
    pts = []
    y = r0
    for i, cnt in enumerate(row_counts):
        shift = r0 if i % 2 == 1 else 0.0
        width = (cnt - 1) * 2 * r0
        x_start = 0.5 - width / 2.0 + shift
        for k in range(cnt):
            if len(pts) >= n: break
            pts.append([x_start + k * 2 * r0, y])
        y += np.sqrt(3) * r0
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing():
    n = 26
    
    # Precompute LP structure
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
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
        
    rng = np.random.default_rng(42)
    best_s = -1.0
    best_c = None
    best_r = None
    
    # Generate diverse initial configurations
    configs = []
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4], 
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [5, 7, 5, 5, 4],
        [6, 5, 5, 6, 4], [5, 6, 6, 4, 5], [7, 6, 6, 7], [8, 6, 6, 6],
        [5, 5, 5, 5, 6], [6, 5, 6, 4, 5], [5, 6, 5, 5, 5], [6, 6, 6, 4, 4]
    ]
    
    for pat in patterns:
        if sum(pat) < n: continue
        r0 = 0.10
        pts = []
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
        pts = np.array(pts[:n])
        pts += rng.uniform(-0.015, 0.015, pts.shape)
        configs.append(pts)
        
    for _ in range(15):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    # Corner-focused inits
    corner_cfg = np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]])
    for _ in range(10):
        cfg = np.concatenate([corner_cfg, rng.uniform(0.2, 0.8, (n-4, 2))])
        configs.append(cfg)
        
    bounds_c = [(0.005, 0.995)] * (2 * n)
    
    # Phase 1: Gradient ascent on centers using LP duals
    for cfg in configs:
        c0 = np.clip(cfg, 0.01, 0.99)
        try:
            res = minimize(obj_func, c0.flatten(), args=(n, pairs, A_ub, m_con), 
                           method='L-BFGS-B', jac=True, bounds=bounds_c, 
                           options={'maxiter': 4000, 'ftol': 1e-13, 'gtol': 1e-12})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(n, 2)
                r_opt, s_opt, _ = solve_lp_and_grad(c_opt, n, pairs, A_ub, m_con)
                if r_opt is not None and s_opt > best_s:
                    best_s = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Perturbation search to escape local minima
    if best_c is not None:
        for _ in range(60):
            c_pert = best_c + rng.uniform(-0.008, 0.008, (n, 2))
            c_pert = np.clip(c_pert, 0.01, 0.99)
            try:
                res_p = minimize(obj_func, c_pert.flatten(), args=(n, pairs, A_ub, m_con),
                                 method='L-BFGS-B', jac=True, bounds=bounds_c,
                                 options={'maxiter': 2000, 'ftol': 1e-13, 'gtol': 1e-12})
                if np.isfinite(res_p.fun):
                    c_p = res_p.x.reshape(n, 2)
                    r_p, s_p, _ = solve_lp_and_grad(c_p, n, pairs, A_ub, m_con)
                    if r_p is not None and s_p > best_s:
                        best_s = s_p
                        best_c = c_p.copy()
                        best_r = r_p.copy()
            except Exception:
                continue
                
    # Phase 3: Joint SLSQP polish for precise boundary/overlap handling
    if best_c is not None:
        v0 = np.concatenate([best_c[:,0], best_c[:,1], best_r * 0.995])
        bounds_slqp = [(0.0, 1.0)] * (2*n) + [(1e-6, 0.5)] * n
        try:
            res_j = minimize(joint_obj, v0, args=(n,), method='SLSQP', bounds=bounds_slqp,
                             constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n,)},
                             options={'maxiter': 5000, 'ftol': 1e-14})
            if np.isfinite(res_j.fun):
                c_j = np.column_stack((res_j.x[:n], res_j.x[n:2*n]))
                r_j, s_j, _ = solve_lp_and_grad(c_j, n, pairs, A_ub, m_con)
                if r_j is not None and s_j > best_s:
                    best_s = s_j
                    best_c = c_j.copy()
                    best_r = r_j.copy()
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
        best_c = np.random.uniform(0.1, 0.9, (n, 2))
        best_r = np.full(n, 0.05)
        best_s = float(np.sum(best_r))
        
    return best_c, best_r, best_s
