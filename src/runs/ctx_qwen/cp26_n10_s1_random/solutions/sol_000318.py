# sol_000318 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000276 (state cc798eee) state=47cfad82 sum of radii=2.332940 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26

def get_tri_indices():
    return np.triu_indices(N, k=1)

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    idx_i, idx_j = get_tri_indices()
    m = len(idx_i)
    
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    diff = centers[idx_i] - centers[idx_j]
    dists = np.sqrt(np.sum(diff**2, axis=1))
    b_ub = dists
    
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    bounds = [(0.0, max(l, 1e-9)) for l in lims]
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun, res
    except Exception:
        pass
    return None, 0.0, None

def compute_gradient(centers, res):
    """Computes gradient of LP sum of radii w.r.t centers using dual variables."""
    n = centers.shape[0]
    idx_i, idx_j = get_tri_indices()
    grad = np.zeros_like(centers)
    
    try:
        marginals = getattr(res, 'marginals', None)
        if marginals is None:
            return grad
            
        lams = getattr(marginals, 'ineqlin', None)
        if lams is None:
            return grad
            
        lams = np.asarray(lams)
        if len(lams) != len(idx_i):
            return grad
            
        active = lams > 1e-6
        if not np.any(active):
            return grad
            
        k = np.where(active)[0]
        i, j = idx_i[k], idx_j[k]
        lam = lams[k]
        
        diff = centers[i] - centers[j]
        dist = np.sqrt(np.sum(diff**2, axis=1))
        dist = np.maximum(dist, 1e-12)
        
        factors = (lam / dist)[:, np.newaxis]
        forces = diff * factors
        
        np.add.at(grad, i, forces)
        np.add.at(grad, j, -forces)
        
    except Exception:
        pass
    return grad

def obj_lp(c_flat):
    """Objective and gradient for L-BFGS-B: maximize LP sum of radii."""
    c = c_flat.reshape(N, 2)
    r, s, res = solve_lp(c)
    if r is None:
        return 1e5, np.zeros_like(c_flat)
    g = compute_gradient(c, res)
    return -s, -g.flatten()

def generate_hex_configs(rng):
    """Generates diverse hexagonal lattice initial configurations."""
    configs = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,5,5], [6,6,5,5,4],
        [5,7,5,5,4], [6,5,5,6,4], [5,6,6,4,5], [7,6,6,7], [8,6,6,6],
        [5,5,5,5,6], [6,5,5,5,5], [5,5,5,6,5], [5,5,6,6,4], [5,6,5,5,5],
        [6,4,5,6,5], [4,5,6,5,6], [5,6,4,6,5], [6,6,6,4,4], [5,6,6,5,4]
    ]
    
    for pat in patterns:
        if sum(pat) < N: continue
        pts = []
        r0 = 0.10
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx % 2 == 1 else 0.0
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
        norm = (base - mn) / span * 0.84 + 0.08
        configs.append(norm)
        
        for _ in range(3):
            p = norm + rng.uniform(-0.02, 0.02, norm.shape)
            configs.append(np.clip(p, 0.04, 0.96))
            
    for _ in range(10):
        configs.append(rng.uniform(0.12, 0.88, (N, 2)))
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_s = -1.0
    best_c = None
    best_r = None
    
    configs = generate_hex_configs(rng)
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    
    # Phase 1: Gradient ascent on centers using LP duals
    for cfg in configs:
        c0 = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(obj_lp, c0.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_c,
                           options={'maxiter': 2500, 'ftol': 1e-13, 'gtol': 1e-9})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp(c_opt)
                if r_opt is not None and s_opt > best_s:
                    best_s = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Stochastic local search to escape local minima
    if best_c is not None:
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_s = best_s
        step = 0.015
        
        for it in range(3000):
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            
            move = rng.uniform(-step, step, 2)
            new_c = np.clip(curr_c[idx] + move, 1e-4, 1.0 - 1e-4)
            curr_c[idx] = new_c
            
            r_try, s_try, _ = solve_lp(curr_c)
            if r_try is not None and s_try > curr_s + 1e-8:
                curr_s = s_try
                curr_r = r_try.copy()
                if curr_s > best_s:
                    best_s = curr_s
                    best_c = curr_c.copy()
                    best_r = curr_r.copy()
                step *= 1.001
            else:
                curr_c[idx] = old
                if rng.random() < 0.02:
                    step *= 0.95
                    
        # Phase 3: L-BFGS-B polish from best stochastic result
        try:
            res = minimize(obj_lp, best_c.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_c,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10})
            if np.isfinite(res.fun):
                c_pol = res.x.reshape(N, 2)
                r_pol, s_pol, _ = solve_lp(c_pol)
                if r_pol is not None and s_pol > best_s:
                    best_s = s_pol
                    best_c = c_pol.copy()
                    best_r = r_pol.copy()
        except Exception:
            pass

    # Fallback
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
                
    best_r *= scale * 0.9999999
    best_s = float(np.sum(best_r))
    
    return best_c, best_r, best_s
