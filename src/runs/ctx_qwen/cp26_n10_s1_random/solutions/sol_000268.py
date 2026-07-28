# sol_000268 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000256 (state fa4faf19) state=670d0c15 sum of radii=2.624511 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def build_indices(n):
    """Precomputes upper triangle indices for pairwise constraints."""
    return np.triu_indices(n, k=1)

def solve_lp(centers, n, triu_i, triu_j):
    """Solves LP to maximize sum of radii for fixed centers."""
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    limits = np.maximum(limits, 1e-9)
    bounds = [(0.0, lim) for lim in limits]
    
    dists = np.hypot(centers[triu_i, 0] - centers[triu_j, 0], 
                     centers[triu_i, 1] - centers[triu_j, 1])
    
    m = len(triu_i)
    A = np.zeros((m, n))
    A[np.arange(m), triu_i] = 1.0
    A[np.arange(m), triu_j] = 1.0
    
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=dists, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def obj_var(v, n):
    """Objective for variable radius optimization."""
    return -np.sum(v[2*n:])

def cons_var(v, n, triu_i, triu_j):
    """Smooth inequality constraints for variable radii."""
    cx, cy, r = v[:n], v[n:2*n], v[2*n:]
    con = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    dx = cx[triu_i] - cx[triu_j]
    dy = cy[triu_i] - cy[triu_j]
    rs = r[triu_i] + r[triu_j]
    con = np.concatenate([con, dx**2 + dy**2 - rs**2])
    return con

def obj_eq(v, n):
    """Objective for equal radius optimization."""
    return -v[2*n]

def cons_eq(v, n, triu_i, triu_j):
    """Smooth inequality constraints for equal radii."""
    cx, cy = v[:n], v[n:2*n]
    t = v[2*n]
    con = np.concatenate([cx - t, 1.0 - cx - t, cy - t, 1.0 - cy - t])
    dx = cx[triu_i] - cx[triu_j]
    dy = cy[triu_i] - cy[triu_j]
    con = np.concatenate([con, dx**2 + dy**2 - 4.0*t**2])
    return con

def gen_hex(n, pat, r0, rng):
    """Generates initial positions on a hexagonal lattice."""
    pts = []
    y = r0
    for idx, cnt in enumerate(pat):
        shift = r0 if idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n: break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
    while len(pts) < n:
        pts.append([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)])
    return np.array(pts[:n])

def run_packing():
    n = 26
    triu_i, triu_j = build_indices(n)
    rng = np.random.default_rng(42)
    
    best_sum = -1.0
    best_c = None
    best_r = None
    
    # Diverse row distributions summing to >= 26
    patterns = [[6,5,6,5,4], [5,6,5,6,4], [6,6,5,5,4], [5,5,6,5,5], [4,6,6,6,4]]
    configs = []
    for p in patterns:
        c = gen_hex(n, p, 0.098, rng)
        configs.append(c)
        cp = c + rng.uniform(-0.02, 0.02, c.shape)
        configs.append(np.clip(cp, 0.05, 0.95))
    for _ in range(10):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    bounds_eq = [(0.0, 1.0)]*(2*n) + [(0.08, 0.12)]
    bounds_var = [(0.0, 1.0)]*(2*n) + [(1e-6, 0.5)]*n
    
    # Phase 1 & 2: Multi-start SLSQP (Equal then Variable radii)
    for cfg in configs:
        # Equal radius phase to find structurally dense centers
        v0_eq = np.concatenate([cfg[:,0], cfg[:,1], [0.10]])
        try:
            res_eq = minimize(obj_eq, v0_eq, args=(n,), method='SLSQP', bounds=bounds_eq,
                              constraints={'type': 'ineq', 'fun': cons_eq, 'args': (n, triu_i, triu_j)},
                              options={'maxiter': 3000, 'ftol': 1e-12})
            if np.isfinite(res_eq.fun):
                c_eq = np.column_stack((res_eq.x[:n], res_eq.x[n:2*n]))
                r_lp, s_lp = solve_lp(c_eq, n, triu_i, triu_j)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_c = c_eq.copy()
                    best_r = r_lp.copy()
        except Exception: pass
        
        # Variable radius phase
        r_init = np.full(n, 0.08)
        v0_var = np.concatenate([cfg[:,0], cfg[:,1], r_init])
        try:
            res_var = minimize(obj_var, v0_var, args=(n,), method='SLSQP', bounds=bounds_var,
                               constraints={'type': 'ineq', 'fun': cons_var, 'args': (n, triu_i, triu_j)},
                               options={'maxiter': 4000, 'ftol': 1e-12})
            if np.isfinite(res_var.fun):
                c_var = np.column_stack((res_var.x[:n], res_var.x[n:2*n]))
                # Refine radii exactly with LP
                r_lp2, s_lp2 = solve_lp(c_var, n, triu_i, triu_j)
                if s_lp2 > best_sum:
                    best_sum = s_lp2
                    best_c = c_var.copy()
                    best_r = r_lp2.copy()
        except Exception: pass

    # Phase 3: LP-Guided Hill Climbing on centers
    if best_c is not None:
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_s = best_sum
        step = 0.035
        
        # Single-circle perturbations
        for it in range(2000):
            idx = rng.integers(n)
            old = curr_c[idx].copy()
            move = rng.uniform(-step, step, 2)
            curr_c[idx] = np.clip(old + move, 1e-4, 1.0-1e-4)
            
            r_try, s_try = solve_lp(curr_c, n, triu_i, triu_j)
            if s_try > curr_s + 1e-9:
                curr_r = r_try
                curr_s = s_try
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_c = curr_c.copy()
                    best_r = curr_r.copy()
                step = min(step * 1.015, 0.06)
            else:
                curr_c[idx] = old
                step *= 0.994
                
        # Multi-circle shakes to escape deep local minima
        for _ in range(50):
            c_shake = best_c.copy()
            idxs = rng.choice(n, size=5, replace=False)
            c_shake[idxs] += rng.uniform(-0.02, 0.02, (5, 2))
            c_shake = np.clip(c_shake, 0.02, 0.98)
            
            r_sh, s_sh = solve_lp(c_shake, n, triu_i, triu_j)
            if s_sh > best_sum:
                best_sum = s_sh
                best_c = c_shake.copy()
                best_r = r_sh.copy()

    # Final strict safety scaling to guarantee validation tolerance
    if best_c is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_c[i,0], best_c[i,1], best_r[i]
            if r > 1e-9:
                scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
        for i in range(n):
            for j in range(i+1, n):
                d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
                rs = best_r[i] + best_r[j]
                if rs > 1e-9:
                    scale = min(scale, d/rs)
                    
        best_r *= scale * 0.9999999
        best_sum = float(np.sum(best_r))
    else:
        # Guaranteed fallback
        best_c = gen_hex(n, [6,5,6,5,4], 0.095, rng)
        best_r, best_sum = solve_lp(best_c, n, triu_i, triu_j)
        
    return best_c, best_r, best_sum
