# sol_000303 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000286 (state b9c01463) state=4cfc8010 sum of radii=0.125225 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_lp_matrix(n):
    pair_i, pair_j = np.triu_indices(n, k=1)
    m_pairs = len(pair_i)
    m_con = 4 * n + m_pairs
    A_ub = np.zeros((m_con, n))
    for i in range(n):
        A_ub[i, i] = 1.0
        A_ub[n + i, i] = 1.0
        A_ub[2*n + i, i] = 1.0
        A_ub[3*n + i, i] = 1.0
    A_ub[4*n:, pair_i] = 1.0
    A_ub[4*n:, pair_j] = 1.0
    return pair_i, pair_j, A_ub, m_con

def solve_lp_and_grad(centers, n, pair_i, pair_j, A_ub, m_con):
    b_ub = np.empty(m_con)
    for i in range(n):
        x, y = centers[i]
        b_ub[i] = x
        b_ub[n + i] = 1.0 - x
        b_ub[2*n + i] = y
        b_ub[3*n + i] = 1.0 - y
        
    dx = centers[pair_i, 0] - centers[pair_j, 0]
    dy = centers[pair_i, 1] - centers[pair_j, 1]
    b_ub[4*n:] = np.hypot(dx, dy)
    
    bounds_r = [(0.0, None)] * n
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success and np.isfinite(res.fun):
            r = res.x
            s = -res.fun
            grad_c = np.zeros((n, 2))
            
            marg = None
            try:
                marg = np.asarray(res.marginals.ineqlin)
            except AttributeError:
                try:
                    marg = np.asarray(res.ineqlin.marginals)
                except AttributeError:
                    pass
                    
            if marg is not None:
                for i in range(n):
                    mu_x = marg[i]
                    mu_1x = marg[n + i]
                    mu_y = marg[2*n + i]
                    mu_1y = marg[3*n + i]
                    grad_c[i, 0] += mu_x - mu_1x
                    grad_c[i, 1] += mu_y - mu_1y
                    
                for k in range(len(pair_i)):
                    lam = marg[4*n + k]
                    if lam > 1e-9:
                        i, j = pair_i[k], pair_j[k]
                        d = b_ub[4*n + k]
                        if d < 1e-12: d = 1e-12
                        fx = lam * (centers[i, 0] - centers[j, 0]) / d
                        fy = lam * (centers[i, 1] - centers[j, 1]) / d
                        grad_c[i, 0] += fx
                        grad_c[i, 1] += fy
                        grad_c[j, 0] -= fx
                        grad_c[j, 1] -= fy
            return r, s, grad_c.flatten()
    except Exception:
        pass
    return None, None, None

def obj_func(c_flat, n, pair_i, pair_j, A_ub, m_con):
    c = c_flat.reshape(n, 2)
    r, s, g = solve_lp_and_grad(c, n, pair_i, pair_j, A_ub, m_con)
    if r is None:
        return 1e6, np.zeros_like(c_flat)
    return -s, -g

def joint_obj(v, n):
    return -np.sum(v[2*n:])

def joint_cons(v, n, triu_idx):
    cx = v[:n]
    cy = v[n:2*n]
    r = v[2*n:]
    cons = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    dx = cx[triu_idx[0]] - cx[triu_idx[1]]
    dy = cy[triu_idx[0]] - cy[triu_idx[1]]
    d2 = dx**2 + dy**2
    rs = r[triu_idx[0]] + r[triu_idx[1]]
    return np.concatenate([cons, d2 - rs**2])

def run_packing():
    n = 26
    pair_i, pair_j, A_ub, m_con = get_lp_matrix(n)
    triu_idx = np.triu_indices(n, k=1)
    rng = np.random.default_rng(42)
    
    best_s = -1.0
    best_c = None
    best_r = None
    
    configs = []
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [6,6,5,5,4], [5,5,6,5,5], [4,6,6,6,4],
                [5,7,5,5,4], [6,5,5,6,4], [5,6,6,4,5], [7,6,6,7], [8,6,6,6],
                [5,5,5,5,6], [6,5,5,5,5], [5,5,5,6,5], [5,5,6,6,4], [5,6,4,6,5]]
                
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
        configs.append(pts)
        for _ in range(3):
            p = pts + rng.uniform(-0.02, 0.02, pts.shape)
            configs.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(25):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * n)
    
    # Phase 1: Gradient ascent on centers using LP duals
    for cfg in configs:
        c0 = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(obj_func, c0.flatten(), args=(n, pair_i, pair_j, A_ub, m_con), 
                           method='L-BFGS-B', jac=True, bounds=bounds_c, 
                           options={'maxiter': 4000, 'ftol': 1e-14, 'gtol': 1e-10})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(n, 2)
                r_opt, s_opt, _ = solve_lp_and_grad(c_opt, n, pair_i, pair_j, A_ub, m_con)
                if r_opt is not None and s_opt > best_s:
                    best_s = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Iterative coordinate perturbation search to escape local minima
    if best_c is not None:
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_s = best_s
        step = 0.015
        for _ in range(3000):
            idx = rng.integers(n)
            old = curr_c[idx].copy()
            move = rng.uniform(-step, step, 2)
            curr_c[idx] = np.clip(old + move, 1e-4, 1.0 - 1e-4)
            
            r_try, s_try, _ = solve_lp_and_grad(curr_c, n, pair_i, pair_j, A_ub, m_con)
            if r_try is not None and s_try > curr_s + 1e-9:
                curr_s = s_try
                curr_r = r_try.copy()
                if rng.random() < 0.5: step *= 1.01
                else: step *= 0.99
            else:
                curr_c[idx] = old
                if rng.random() < 0.02: step *= 0.95
                
        best_c = curr_c
        best_r = curr_r
        best_s = curr_s

    # Phase 3: Multi-scale perturbation + Gradient ascent
    if best_c is not None:
        for scale in [0.008, 0.003, 0.001]:
            for _ in range(20):
                c_pert = best_c + rng.uniform(-scale, scale, (n, 2))
                c_pert = np.clip(c_pert, 1e-4, 1.0 - 1e-4)
                try:
                    res_p = minimize(obj_func, c_pert.flatten(), args=(n, pair_i, pair_j, A_ub, m_con),
                                     method='L-BFGS-B', jac=True, bounds=bounds_c,
                                     options={'maxiter': 2000, 'ftol': 1e-13})
                    if np.isfinite(res_p.fun):
                        c_p = res_p.x.reshape(n, 2)
                        r_p, s_p, _ = solve_lp_and_grad(c_p, n, pair_i, pair_j, A_ub, m_con)
                        if r_p is not None and s_p > best_s:
                            best_s = s_p
                            best_c = c_p.copy()
                            best_r = r_p.copy()
                except Exception:
                    continue
                    
    # Phase 4: Joint SLSQP polish for precise boundary/overlap handling
    if best_c is not None:
        v0 = np.concatenate([best_c[:,0], best_c[:,1], best_r * 0.995])
        bounds_slqp = [(0.0, 1.0)] * (2*n) + [(1e-6, 0.5)] * n
        try:
            res_j = minimize(joint_obj, v0, args=(n,), method='SLSQP', bounds=bounds_slqp,
                             constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n, triu_idx)},
                             options={'maxiter': 5000, 'ftol': 1e-14})
            if np.isfinite(res_j.fun):
                c_j = np.column_stack((res_j.x[:n], res_j.x[n:2*n]))
                r_j, s_j, _ = solve_lp_and_grad(c_j, n, pair_i, pair_j, A_ub, m_con)
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
        best_c = rng.uniform(0.1, 0.9, (n, 2))
        best_r = np.full(n, 0.05)
        best_s = float(np.sum(best_r))
        
    return best_c, best_r, best_s
