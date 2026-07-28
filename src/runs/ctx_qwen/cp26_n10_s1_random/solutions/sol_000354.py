# sol_000354 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000225 (state c5495767) state=a5a54396 sum of radii=2.607906 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
triu_i, triu_j = np.triu_indices(N, k=1)
M_PAIRS = len(triu_i)
M_CON = M_PAIRS + 4 * N

# Precompute constant LP constraint matrix structure
A_LP = np.zeros((M_CON, N))
for k, (i, j) in enumerate(zip(triu_i, triu_j)):
    A_LP[k, i] = 1.0
    A_LP[k, j] = 1.0
for i in range(N):
    A_LP[M_PAIRS + i, i] = 1.0
    A_LP[M_PAIRS + N + i, i] = 1.0
    A_LP[M_PAIRS + 2*N + i, i] = 1.0
    A_LP[M_PAIRS + 3*N + i, i] = 1.0

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-8)
    bounds = [(0.0, lim) for lim in lims]
    
    b_ub = np.zeros(M_CON)
    diffs = centers[triu_i] - centers[triu_j]
    b_ub[:M_PAIRS] = np.hypot(diffs[:, 0], diffs[:, 1])
    
    for i in range(N):
        x, y = centers[i]
        b_ub[M_PAIRS + i] = x
        b_ub[M_PAIRS + N + i] = 1.0 - x
        b_ub[M_PAIRS + 2*N + i] = y
        b_ub[M_PAIRS + 3*N + i] = 1.0 - y
        
    try:
        res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun, res
    except Exception:
        pass
    return None, 0.0, None

def get_grad(centers, res):
    """Computes gradient of sum of radii w.r.t centers using LP duals."""
    grad = np.zeros((N, 2))
    if res is None:
        return grad
    try:
        marg = np.asarray(res.marginals.ineqlin)
    except Exception:
        return grad
        
    lams = marg[:M_PAIRS]
    mask = lams > 1e-7
    if np.any(mask):
        idx = np.where(mask)[0]
        lam_vals = lams[idx]
        ii = triu_i[idx]
        jj = triu_j[idx]
        dx = centers[ii, 0] - centers[jj, 0]
        dy = centers[ii, 1] - centers[jj, 1]
        d = np.sqrt(dx**2 + dy**2)
        d = np.where(d < 1e-12, 1e-12, d)
        fx = lam_vals * dx / d
        fy = lam_vals * dy / d
        
        np.add.at(grad[:, 0], ii, fx)
        np.add.at(grad[:, 1], ii, fy)
        np.add.at(grad[:, 0], jj, -fx)
        np.add.at(grad[:, 1], jj, -fy)
        
    for i in range(N):
        grad[i, 0] += marg[M_PAIRS + i] - marg[M_PAIRS + N + i]
        grad[i, 1] += marg[M_PAIRS + 2*N + i] - marg[M_PAIRS + 3*N + i]
        
    return grad

def obj_grad(c_flat):
    """Objective and gradient for center optimization."""
    c = c_flat.reshape(N, 2)
    r, s, res = solve_lp(c)
    if r is None:
        return 1e6, np.zeros_like(c_flat)
    grad = get_grad(c, res)
    return -s, -grad.flatten()

def slsqp_obj(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2*N:])

def slsqp_cons(v):
    """Inequality constraints >= 0 for SLSQP."""
    cx = v[:N]
    cy = v[N:2*N]
    r = v[2*N:]
    
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c = np.concatenate([c, d2[mask] - rs[mask]**2])
    return c

def run_packing():
    rng = np.random.default_rng(42)
    best_s = -1.0
    best_c = None
    best_r = None
    
    # Generate diverse initial configurations
    configs = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [6, 5, 5, 6, 4], [5, 6, 6, 4, 5], [4, 5, 6, 5, 6]
    ]
    
    for pat in patterns:
        if sum(pat) < N: continue
        r0 = 0.10
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
        norm_base = (base - mn) / span * 0.86 + 0.07
        configs.append(norm_base)
        for _ in range(4):
            p = norm_base + rng.uniform(-0.02, 0.02, norm_base.shape)
            configs.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(10):
        configs.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    
    # Phase 1: Gradient ascent on centers using LP duals
    for cfg in configs:
        c0 = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(obj_grad, c0.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_c,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'gtol': 1e-10})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp(c_opt)
                if r_opt is not None and s_opt > best_s:
                    best_s = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Local coordinate descent / jiggle search
    if best_c is not None:
        curr_c = best_c.copy()
        step = 0.015
        for it in range(3000):
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            move = rng.uniform(-step, step, 2)
            new_c = np.clip(curr_c[idx] + move, 1e-4, 1.0 - 1e-4)
            curr_c[idx] = new_c
            
            r_try, s_try, _ = solve_lp(curr_c)
            if r_try is not None and s_try > best_s + 1e-9:
                best_s = s_try
                best_c = curr_c.copy()
                best_r = r_try.copy()
                step = min(step * 1.002, 0.05)
            else:
                curr_c[idx] = old
                step *= 0.998
                
    # Phase 3: Joint SLSQP polish for precise constraint handling
    if best_c is not None:
        for _ in range(8):
            c_pert = np.clip(best_c + rng.uniform(-0.005, 0.005, best_c.shape), 1e-4, 1.0-1e-4)
            r_pert = best_r * 0.99
            v0 = np.concatenate([c_pert[:, 0], c_pert[:, 1], r_pert])
            bounds_slqp = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
            
            try:
                res_j = minimize(slsqp_obj, v0, method='SLSQP', bounds=bounds_slqp,
                                 constraints={'type': 'ineq', 'fun': slsqp_cons},
                                 options={'maxiter': 4000, 'ftol': 1e-13})
                if np.isfinite(res_j.fun):
                    c_j = np.column_stack((res_j.x[:N], res_j.x[N:2*N]))
                    r_j, s_j, _ = solve_lp(c_j)
                    if r_j is not None and s_j > best_s:
                        best_s = s_j
                        best_c = c_j.copy()
                        best_r = r_j.copy()
            except Exception:
                continue
                
    # Fallback safety net
    if best_c is None:
        best_c = configs[0]
        best_r, best_s, _ = solve_lp(best_c)
        
    # Final strict safety scaling to guarantee numerical validity against 1e-12 tolerance
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
                
    best_r *= scale * 0.999999
    best_s = float(np.sum(best_r))
    
    return best_c, best_r, best_s
