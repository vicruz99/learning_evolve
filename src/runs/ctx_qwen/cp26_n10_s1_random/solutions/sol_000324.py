# sol_000324 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000308 (state b85a4809) state=1f79692d sum of radii=2.598982 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
idx_i, idx_j = np.triu_indices(N, k=1)
M_PAIRS = len(idx_i)
A_ub = np.zeros((M_PAIRS, N))
for k, (i, j) in enumerate(zip(idx_i, idx_j)):
    A_ub[k, i] = 1.0
    A_ub[k, j] = 1.0

def solve_lp(centers):
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, lim) for lim in lims]
    
    dx = centers[idx_i, 0] - centers[idx_j, 0]
    dy = centers[idx_i, 1] - centers[idx_j, 1]
    dists = np.hypot(dx, dy)
    dists = np.maximum(dists, 1e-9)
    
    try:
        res = linprog(-np.ones(N), A_ub=A_ub, b_ub=dists, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun, res
    except Exception:
        pass
    return None, 0.0, None

def obj_grad(c_flat):
    c = c_flat.reshape(N, 2)
    r, s, res = solve_lp(c)
    if r is None:
        return 1e6, np.zeros_like(c_flat)
        
    grad = np.zeros((N, 2))
    marg = None
    try:
        if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
            marg = np.asarray(res.marginals.ineqlin)
        elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
            marg = np.asarray(res.ineqlin.marginals)
    except Exception:
        pass
        
    if marg is not None:
        lams = marg
        lams_p = lams[:M_PAIRS]
        mask = lams_p > 1e-9
        if np.any(mask):
            idx = np.where(mask)[0]
            lam_vals = lams_p[idx]
            ii = idx_i[idx]
            jj = idx_j[idx]
            dx = c[ii, 0] - c[jj, 0]
            dy = c[ii, 1] - c[jj, 1]
            d = np.hypot(dx, dy)
            d = np.where(d < 1e-12, 1e-12, d)
            fx = lam_vals * dx / d
            fy = lam_vals * dy / d
            np.add.at(grad[:, 0], ii, fx)
            np.add.at(grad[:, 1], ii, fy)
            np.add.at(grad[:, 0], jj, -fx)
            np.add.at(grad[:, 1], jj, -fy)
        return -s, -grad.flatten()
    return -s, np.zeros_like(c_flat)

def obj_func(c_flat):
    return obj_grad(c_flat)[0]

def jac_func(c_flat):
    return obj_grad(c_flat)[1]

def joint_obj(v):
    return -np.sum(v[2*N:])

def joint_cons(v):
    cx, cy, r = v[:N], v[N:2*N], v[2*N:]
    bc = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    dx = cx[idx_i] - cx[idx_j]
    dy = cy[idx_i] - cy[idx_j]
    d2 = dx**2 + dy**2
    rs = r[idx_i] + r[idx_j]
    pc = d2 - rs**2
    return np.concatenate([bc, pc])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_s = -1.0
    best_c = None
    best_r = None
    
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4], [6,6,5,5,4],
        [5,7,5,5,4], [6,5,5,6,4], [5,6,6,4,5], [7,6,6,7], [5,5,5,5,6],
        [6,5,5,5,5], [5,5,5,6,5], [5,5,6,6,4], [5,6,5,5,5], [5,4,6,6,5],
        [6,4,5,6,5], [4,5,6,5,6], [7,5,6,6], [6,6,6,6,2], [5,5,5,7,4]
    ]
    
    configs = []
    for pat in patterns:
        if sum(pat) < N: continue
        r0 = 0.095
        pts = []
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= N: break
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
        base = np.array(pts[:N])
        mn = base.min(axis=0)
        mx = base.max(axis=0)
        span = mx - mn + 1e-9
        norm_base = (base - mn) / span * 0.88 + 0.06
        configs.append(norm_base)
        
        for _ in range(5):
            p = norm_base + rng.uniform(-0.035, 0.035, norm_base.shape)
            configs.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(12):
        configs.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    
    for cfg in configs:
        c0 = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(obj_func, c0.flatten(), method='L-BFGS-B',
                           jac=jac_func, bounds=bounds_c,
                           options={'maxiter': 6000, 'ftol': 1e-14, 'gtol': 1e-10})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp(c_opt)
                if r_opt is not None and s_opt > best_s:
                    best_s = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            continue
            
    if best_c is not None:
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_s = best_s
        step = 0.014
        
        for it in range(4000):
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            new_c = np.clip(curr_c[idx] + rng.uniform(-step, step, 2), 1e-4, 1.0 - 1e-4)
            curr_c[idx] = new_c
            
            r_try, s_try, _ = solve_lp(curr_c)
            if r_try is not None and s_try > curr_s + 1e-8:
                curr_s = s_try
                curr_r = r_try.copy()
                if curr_s > best_s:
                    best_s = curr_s
                    best_c = curr_c.copy()
                    best_r = curr_r.copy()
                step *= 0.9975
            else:
                curr_c[idx] = old
                if rng.random() < 0.04:
                    step *= 0.94
                    
    if best_c is not None:
        for _ in range(10):
            c_pert = np.clip(best_c + rng.uniform(-0.006, 0.006, best_c.shape), 1e-4, 1.0-1e-4)
            r_pert = best_r * 0.995
            v0 = np.concatenate([c_pert[:, 0], c_pert[:, 1], r_pert])
            bounds_slqp = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
            
            try:
                res_j = minimize(joint_obj, v0, method='SLSQP', bounds=bounds_slqp,
                                 constraints={'type': 'ineq', 'fun': joint_cons},
                                 options={'maxiter': 5000, 'ftol': 1e-14})
                if np.isfinite(res_j.fun):
                    c_j = np.column_stack((res_j.x[:N], res_j.x[N:2*N]))
                    r_j, s_j, _ = solve_lp(c_j)
                    if r_j is not None and s_j > best_s:
                        best_s = s_j
                        best_c = c_j.copy()
                        best_r = r_j.copy()
            except Exception:
                continue
                
    if best_c is None:
        best_c = configs[0]
        best_r, best_s, _ = solve_lp(best_c)
        
    scale = 1.0
    for i in range(N):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_r *= scale * 0.9999995
    best_s = float(np.sum(best_r))
    
    return best_c, best_r, best_s
