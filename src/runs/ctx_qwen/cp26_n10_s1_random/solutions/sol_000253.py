# sol_000253 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000227 (state bd5d11f3) state=86316387 sum of radii=2.626165 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp(centers, n):
    """Solves LP to maximize sum of radii for fixed centers."""
    m_b = 4 * n
    m_p = n * (n - 1) // 2
    A = np.zeros((m_b + m_p, n))
    b = np.zeros(m_b + m_p)
    idx = 0
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        lims = (x, 1.0 - x, y, 1.0 - y)
        for lim in lims:
            A[idx, i] = 1.0
            b[idx] = lim
            idx += 1
            
    # Pairwise constraints: r_i + r_j <= dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            b[idx] = d
            idx += 1
            
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=[(0, None)] * n, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except:
        pass
    return None, 0.0

def cons_slsqp(v, n, ii, jj):
    """Inequality constraints >= 0 for SLSQP."""
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    dx = x[ii] - x[jj]
    dy = y[ii] - y[jj]
    dr = r[ii] + r[jj]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def obj_slsqp(v, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def run_packing():
    n = 26
    ii, jj = np.triu_indices(n, k=1)
    best_sum = -1.0
    best_c = None
    best_r = None
    
    rng = np.random.default_rng(42)
    configs = []
    
    # Diverse hexagonal row distributions summing to ~26
    row_dists = [
        [6,5,6,5,4], [5,6,5,6,4], [7,6,6,7], [8,6,6,6], 
        [9,5,6,6], [6,6,5,5,4], [5,5,6,5,5], [4,6,6,6,4],
        [6,5,5,5,5], [5,5,5,5,6], [10,8,8], [9,9,8]
    ]
    
    for rd in row_dists:
        if sum(rd) < n: 
            continue
        for _ in range(2):
            r0 = rng.uniform(0.092, 0.102)
            pts = []
            y = r0
            for idx, cnt in enumerate(rd):
                shift = r0 if idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(pts) >= n: 
                        break
                    pts.append([x, y])
                    x += 2.0 * r0
                y += np.sqrt(3) * r0
            cfg = np.array(pts[:n])
            # Normalize and shift to [0.1, 0.9] with small random jitter
            mn, mx = cfg.min(axis=0), cfg.max(axis=0)
            cfg = (cfg - mn) / (mx - mn + 1e-9) * 0.75 + 0.125
            cfg += rng.uniform(-0.015, 0.015, cfg.shape)
            cfg = np.clip(cfg, 0.03, 0.97)
            configs.append(cfg)
            
    # Add purely random starts
    for _ in range(5):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # Phase 1: SLSQP optimization from diverse starts
    for cfg in configs:
        lims = np.min(np.stack([cfg[:,0], 1.0-cfg[:,0], cfg[:,1], 1.0-cfg[:,1]], axis=1), axis=1)
        diffs = cfg[:, None, :] - cfg[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists, axis=1) / 2.0
        r_init = np.minimum(lims, min_d) * 0.95
        
        v0 = np.concatenate([cfg[:,0], cfg[:,1], r_init])
        
        try:
            res = minimize(obj_slsqp, v0, args=(n,), method='SLSQP',
                           bounds=bounds,
                           constraints={'type': 'ineq', 'fun': cons_slsqp, 'args': (n, ii, jj)},
                           options={'maxiter': 6000, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                c_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
                r_lp, s_lp = solve_lp(c_opt, n)
                if r_lp is not None and s_lp > best_sum:
                    best_sum = s_lp
                    best_c = c_opt.copy()
                    best_r = r_lp.copy()
        except:
            continue

    # Phase 2: Coordinate-wise hill climbing on centers using LP objective
    if best_c is not None:
        curr_c = best_c.copy()
        curr_s = best_sum
        step = 0.035
        
        for iteration in range(3000):
            idx = rng.integers(n)
            old = curr_c[idx].copy()
            curr_c[idx] += rng.uniform(-step, step, 2)
            curr_c[idx] = np.clip(curr_c[idx], 0.01, 0.99)
            
            r_try, s_try = solve_lp(curr_c, n)
            if r_try is not None and s_try > curr_s + 1e-9:
                curr_s = s_try
                if s_try > best_sum:
                    best_sum = s_try
                    best_c = curr_c.copy()
                    best_r = r_try.copy()
            else:
                curr_c[idx] = old
            step *= 0.997
            
        # Phase 3: Final SLSQP polish from improved centers
        v0_pol = np.concatenate([best_c[:,0], best_c[:,1], best_r * 0.98])
        try:
            res_p = minimize(obj_slsqp, v0_pol, args=(n,), method='SLSQP',
                             bounds=bounds,
                             constraints={'type': 'ineq', 'fun': cons_slsqp, 'args': (n, ii, jj)},
                             options={'maxiter': 6000, 'ftol': 1e-14})
            if np.isfinite(res_p.fun):
                c_p = np.column_stack((res_p.x[:n], res_p.x[n:2*n]))
                r_p, s_p = solve_lp(c_p, n)
                if r_p is not None and s_p > best_sum:
                    best_c = c_p
                    best_r = r_p
                    best_sum = s_p
        except:
            pass
            
    # Fallback if all optimizations fail
    if best_c is None:
        best_c = configs[0]
        best_r, best_sum = solve_lp(best_c, n)
        if best_r is None: 
            best_r = np.full(n, 0.08)
            
    # Phase 4: Strict safety scaling to guarantee numerical validity
    scale = 1.0
    for i in range(n):
        x, y, r = best_c[i,0], best_c[i,1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
            
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_r *= scale * 0.9999999
    return best_c, best_r, float(np.sum(best_r))
