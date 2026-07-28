# sol_000180 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000118 (state b8add980) state=c7dec4dc sum of radii=2.614875 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def obj_equal(vars_arr):
    """Objective for equal-radius optimization: maximize t => minimize -t."""
    return -vars_arr[-1]

def get_constraints_equal(vars_arr, n):
    """Returns inequality constraints >= 0 for equal-radius valid packing."""
    c = vars_arr[:2*n].reshape(n, 2)
    t = vars_arr[2*n]
    cons = []
    
    # Boundary constraints
    cons.append(c[:, 0] - t)
    cons.append(1.0 - c[:, 0] - t)
    cons.append(c[:, 1] - t)
    cons.append(1.0 - c[:, 1] - t)
    
    # Pairwise non-overlap: dist^2 >= (2t)^2
    dx = c[:, 0:1] - c[:, 0:1].T
    dy = c[:, 1:2] - c[:, 1:2].T
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)
    
    r_sum = 2.0 * t
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    cons.append(d2[mask] - r_sum**2)
    return np.concatenate(cons)

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    limits = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                        np.minimum(centers[:, 1], 1 - centers[:, 1]))
    limits = np.maximum(limits, 1e-9)
    
    c_obj = -np.ones(n)
    bounds = [(0.0, lim) for lim in limits]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1e9)
    
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def joint_obj(vars_arr, n):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(vars_arr[2*n:])

def joint_cons(vars_arr, n):
    """Returns inequality constraints >= 0 for variable-radius valid packing."""
    c = vars_arr[:2*n].reshape(n, 2)
    r = vars_arr[2*n:]
    cons = [c[:, 0] - r, 1.0 - c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 1] - r]
    
    dx = c[:, 0:1] - c[:, 0:1].T
    dy = c[:, 1:2] - c[:, 1:2].T
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    cons.append(d2[mask] - r_sum[mask]**2)
    return np.concatenate(cons)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Competitive row distributions for N=26 hexagonal packing
    row_patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [5, 6, 4, 6, 5], [5, 4, 6, 5, 6],
        [6, 4, 5, 6, 5], [5, 5, 5, 5, 6], [6, 5, 5, 5, 5],
        [5, 7, 4, 5, 5], [4, 5, 6, 6, 5]
    ]
    
    bounds_eq = [(0.0, 1.0)] * (2*n) + [(0.05, 0.12)]
    bounds_jt = [(0.0, 1.0)] * (2*n) + [(1e-4, 0.5)] * n
    
    def make_hex(pat, r0):
        pts = []
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) < n: pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
        pts = np.array(pts[:n])
        # Center and scale to fit comfortably inside [0.1, 0.9]
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = mx - mn
        if span[0] > 0: pts[:, 0] = (pts[:, 0] - mn[0]) / span[0] * 0.8 + 0.1
        if span[1] > 0: pts[:, 1] = (pts[:, 1] - mn[1]) / span[1] * 0.8 + 0.1
        return pts

    rng = np.random.default_rng(42)
    configs = []
    for pat in row_patterns:
        if sum(pat) != n: continue
        c = make_hex(pat, 0.1)
        configs.append(c)
        for _ in range(4):
            p = c + rng.uniform(-0.02, 0.02, c.shape)
            configs.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(5):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))

    # Phase 1: Optimize centers for maximal equal radius
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), [0.08]])
        try:
            res = minimize(obj_equal, x0, method='SLSQP', bounds=bounds_eq,
                          constraints={'type': 'ineq', 'fun': get_constraints_equal, 'args': (n,)},
                          options={'maxiter': 6000, 'ftol': 1e-12})
            if np.isfinite(res.fun):
                c_opt = res.x[:2*n].reshape(n, 2)
                # LP refinement guarantees optimal radii for these centers
                r_lp, s_lp = solve_lp_radii(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception: pass

    # Phase 2: Joint optimization to deform configuration and maximize sum
    if best_centers is not None:
        for _ in range(10):
            pert = best_centers + rng.uniform(-0.005, 0.005, best_centers.shape)
            pert = np.clip(pert, 0.05, 0.95)
            r0 = best_radii * 0.96
            x0 = np.concatenate([pert.flatten(), r0])
            try:
                res = minimize(joint_obj, x0, args=(n,), method='SLSQP', bounds=bounds_jt,
                              constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n,)},
                              options={'maxiter': 6000, 'ftol': 1e-12})
                if np.isfinite(res.fun):
                    c_opt = res.x[:2*n].reshape(n, 2)
                    r_opt = res.x[2*n:]
                    cons_val = joint_cons(res.x, n)
                    if np.min(cons_val) > -1e-5:
                        s = np.sum(r_opt)
                        if s > best_sum:
                            best_sum = s
                            best_centers = c_opt.copy()
                            best_radii = r_opt.copy()
            except Exception: pass

    # Fallback safety net
    if best_centers is None:
        best_centers = make_hex([5, 6, 5, 6, 4], 0.1)
        best_radii, best_sum = solve_lp_radii(best_centers)

    # Final safety scaling to strictly satisfy 1e-12 validator tolerance
    scale = 1.0
    c = best_centers
    r = best_radii
    for i in range(n):
        if r[i] > 1e-12:
            scale = min(scale, c[i,0]/r[i], (1-c[i,0])/r[i], c[i,1]/r[i], (1-c[i,1])/r[i])
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(c[i]-c[j])
            r_sum = r[i] + r[j]
            if r_sum > 1e-12:
                scale = min(scale, d / r_sum)
                
    r *= scale * 0.9999995
    best_sum = float(np.sum(r))
    best_radii = r
    
    return best_centers, best_radii, best_sum
