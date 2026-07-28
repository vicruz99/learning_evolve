# sol_000309 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000286 (state b9c01463) state=79ada07f sum of radii=2.288016 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

def solve_lp_and_grad(centers, n, pair_i, pair_j, A_lp):
    """Solves LP for radii given centers and computes gradient of sum of radii w.r.t centers."""
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, lim) for lim in lims]
    
    diff = centers[pair_i] - centers[pair_j]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    
    try:
        res = linprog(-np.ones(n), A_ub=A_lp, b_ub=dists, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            r = res.x
            s = -res.fun
            grad_c = np.zeros((n, 2))
            try:
                marg = np.asarray(res.marginals.ineqlin)
            except:
                marg = None
            if marg is not None:
                active = marg > 1e-9
                if np.any(active):
                    lam = marg[active]
                    pi, pj = pair_i[active], pair_j[active]
                    d_vec = centers[pi] - centers[pj]
                    d_norm = np.sqrt(np.sum(d_vec**2, axis=1))
                    d_norm = np.where(d_norm < 1e-12, 1e-12, d_norm)
                    inv_d = lam / d_norm[:, np.newaxis]
                    grad_c[pi] += d_vec * inv_d
                    grad_c[pj] -= d_vec * inv_d
            return r, s, grad_c.flatten()
    except:
        pass
    return None, None, None

def obj_lp(c_flat, n, pair_i, pair_j, A_lp):
    """Objective function for L-BFGS-B: negative sum of radii and its gradient."""
    c = c_flat.reshape(n, 2)
    r, s, g = solve_lp_and_grad(c, n, pair_i, pair_j, A_lp)
    if r is None:
        return 1e6, np.zeros_like(c_flat)
    return -s, -g

def joint_obj(v, n):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def joint_cons(v, n, pair_i, pair_j):
    """Inequality constraints >= 0 for valid packing in SLSQP."""
    cx, cy, r = v[:n], v[n:2*n], v[2*n:]
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    dx = cx[pair_i] - cx[pair_j]
    dy = cy[pair_i] - cy[pair_j]
    d2 = dx**2 + dy**2
    rs = r[pair_i] + r[pair_j]
    c = np.concatenate([c, d2 - rs**2])
    return c

def run_packing():
    n = 26
    pair_i, pair_j = np.triu_indices(n, k=1)
    n_pairs = len(pair_i)
    A_lp = np.zeros((n_pairs, n))
    A_lp[np.arange(n_pairs), pair_i] = 1.0
    A_lp[np.arange(n_pairs), pair_j] = 1.0
    
    rng = np.random.default_rng(42)
    best_s = -1.0
    best_c = None
    best_r = None
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * n)
    
    configs = []
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4],
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [5, 7, 5, 5, 4],
        [6, 5, 5, 6, 4], [5, 6, 6, 4, 5], [7, 6, 6, 7], [8, 6, 6, 6]
    ]
    
    for pat in patterns:
        if sum(pat) < n: continue
        r0 = 0.095
        pts = []
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 2.0 * r0
            y += math.sqrt(3.0) * r0
        cfg = np.array(pts[:n])
        mn, mx = cfg.min(axis=0), cfg.max(axis=0)
        cfg = (cfg - mn) / (mx - mn) * 0.85 + 0.075
        
        configs.append(cfg)
        for angle in [0.1, -0.1, 0.2, -0.2]:
            cos_t, sin_t = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            cfg_rot = cfg @ rot
            mn_r, mx_r = cfg_rot.min(axis=0), cfg_rot.max(axis=0)
            cfg_rot = (cfg_rot - mn_r) / (mx_r - mn_r) * 0.85 + 0.075
            configs.append(cfg_rot)
            
        for _ in range(3):
            configs.append(np.clip(cfg + rng.uniform(-0.03, 0.03, cfg.shape), 0.02, 0.98))
            
    for _ in range(15):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    # Phase 1: Gradient ascent on centers using LP duals
    for cfg in configs:
        c0 = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(obj_lp, c0.flatten(), args=(n, pair_i, pair_j, A_lp),
                           jac=True, method='L-BFGS-B', bounds=bounds_c,
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(n, 2)
                r_opt, s_opt, _ = solve_lp_and_grad(c_opt, n, pair_i, pair_j, A_lp)
                if r_opt is not None and s_opt > best_s:
                    best_s = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except:
            continue
            
    # Phase 2: Iterative perturbation search to escape local minima
    if best_c is not None:
        step = 0.005
        for _ in range(300):
            idx = rng.integers(n)
            old = best_c[idx].copy()
            best_c[idx] += rng.uniform(-step, step, 2)
            best_c[idx] = np.clip(best_c[idx], 1e-4, 1.0 - 1e-4)
            
            r_try, s_try, _ = solve_lp_and_grad(best_c, n, pair_i, pair_j, A_lp)
            if r_try is not None and s_try > best_s:
                best_s = s_try
                best_r = r_try.copy()
                try:
                    res_loc = minimize(obj_lp, best_c.flatten(), args=(n, pair_i, pair_j, A_lp),
                                       jac=True, method='L-BFGS-B', bounds=bounds_c,
                                       options={'maxiter': 1500, 'ftol': 1e-13})
                    if np.isfinite(res_loc.fun):
                        c_loc = res_loc.x.reshape(n, 2)
                        r_loc, s_loc, _ = solve_lp_and_grad(c_loc, n, pair_i, pair_j, A_lp)
                        if r_loc is not None and s_loc > best_s:
                            best_s = s_loc
                            best_c = c_loc.copy()
                            best_r = r_loc.copy()
                except:
                    pass
            else:
                best_c[idx] = old
            step *= 0.995
            
    # Phase 3: Joint SLSQP polish for precise boundary/overlap handling
    if best_c is not None:
        bounds_slqp = [(0.0, 1.0)] * (2*n) + [(1e-6, 0.5)] * n
        for _ in range(5):
            v0 = np.concatenate([best_c[:, 0], best_c[:, 1], best_r * 0.999])
            v0[:2*n] += rng.uniform(-0.002, 0.002, 2*n)
            v0[:2*n] = np.clip(v0[:2*n], 1e-4, 1.0 - 1e-4)
            try:
                res_slqp = minimize(joint_obj, v0, args=(n,), method='SLSQP', bounds=bounds_slqp,
                                    constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n, pair_i, pair_j)},
                                    options={'maxiter': 5000, 'ftol': 1e-14})
                if np.isfinite(res_slqp.fun):
                    c_slqp = res_slqp.x[:2*n].reshape(n, 2)
                    r_slqp, s_slqp, _ = solve_lp_and_grad(c_slqp, n, pair_i, pair_j, A_lp)
                    if r_slqp is not None and s_slqp > best_s:
                        best_s = s_slqp
                        best_c = c_slqp.copy()
                        best_r = r_slqp.copy()
            except:
                pass
                
    # Final strict safety scaling to guarantee numerical validity
    if best_c is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
            if r > 1e-12:
                scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
                rs = best_r[i] + best_r[j]
                if rs > 1e-12:
                    scale = min(scale, d/rs)
        best_r *= scale * 0.9999998
        best_s = float(np.sum(best_r))
    else:
        best_c = rng.uniform(0.2, 0.8, (n, 2))
        best_r = np.full(n, 0.08)
        best_s = float(np.sum(best_r))
        
    return best_c, best_r, best_s
