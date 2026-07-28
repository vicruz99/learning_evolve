# sol_000140 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000067 (state 3fcdd2a7) state=07ed95ff sum of radii=2.628410 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_radii_lp(centers, n, idx_i, idx_j, limits):
    """Solves LP to maximize sum of radii for fixed centers."""
    c_obj = -np.ones(n)
    bounds = [(0.0, max(0.0, lim)) for lim in limits]
    
    A_ub = np.zeros((len(idx_i), n))
    A_ub[np.arange(len(idx_i)), idx_i] = 1.0
    A_ub[np.arange(len(idx_i)), idx_j] = 1.0
    
    diffs = centers[idx_i] - centers[idx_j]
    b_ub = np.sqrt(np.sum(diffs**2, axis=1))
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def obj_func(vars, n):
    return -np.sum(vars[2*n:3*n])

def con_func(vars, n, idx_i, idx_j):
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:3*n]
    bc = np.concatenate([c[:, 0] - r, 1 - c[:, 0] - r, c[:, 1] - r, 1 - c[:, 1] - r])
    diffs = c[idx_i] - c[idx_j]
    dist_sq = np.sum(diffs**2, axis=1)
    r_sum = r[idx_i] + r[idx_j]
    pc = dist_sq - r_sum**2
    return np.concatenate([bc, pc])

def get_boundary_limits(c):
    return np.maximum(np.minimum(np.minimum(c[:, 0], 1 - c[:, 0]), 
                                 np.minimum(c[:, 1], 1 - c[:, 1])), 0.0)

def run_packing():
    n = 26
    idx_i, idx_j = np.triu_indices(n, k=1)
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds_vars = [(0.0, 1.0)] * (2*n) + [(1e-6, 0.5)] * n
    
    configs = []
    for r0 in [0.08, 0.085, 0.09, 0.095, 0.10]:
        pts = []
        row = 0
        y = r0
        while len(pts) < n:
            x = r0 + (row % 2) * r0
            while x + r0 <= 1.0 and len(pts) < n:
                pts.append([x, y])
                x += 2.0 * r0
            if len(pts) < n:
                y += np.sqrt(3) * r0
                row += 1
        configs.append(np.array(pts[:n]))
        
    for base in configs[:5]:
        for _ in range(3):
            p = base + rng.uniform(-0.03, 0.03, (n, 2))
            configs.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(10):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    for cfg in configs:
        limits = get_boundary_limits(cfg)
        r0, _ = solve_radii_lp(cfg, n, idx_i, idx_j, limits)
        if np.sum(r0) < 1.0:
            r0 = np.full(n, 0.05)
            
        x0 = np.concatenate([cfg.flatten(), r0])
        
        try:
            res = minimize(obj_func, x0, args=(n,), method='SLSQP', bounds=bounds_vars, 
                          constraints={'type': 'ineq', 'fun': con_func, 'args': (n, idx_i, idx_j)},
                          options={'maxiter': 8000, 'ftol': 1e-13})
            
            if np.isfinite(res.fun):
                c_opt = res.x[:2*n].reshape(n, 2)
                r_opt = res.x[2*n:3*n]
                
                c_vals = con_func(res.x, n, idx_i, idx_j)
                if np.min(c_vals) >= -1e-7:
                    limits_opt = get_boundary_limits(c_opt)
                    r_lp, s_lp = solve_radii_lp(c_opt, n, idx_i, idx_j, limits_opt)
                    if s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = c_opt.copy()
                        best_radii = r_lp.copy()
        except Exception:
            continue
            
    if best_centers is not None:
        for _ in range(200):
            c_pert = best_centers + rng.normal(0, 0.004, (n, 2))
            c_pert = np.clip(c_pert, 0.01, 0.99)
            limits_p = get_boundary_limits(c_pert)
            r_lp, s_lp = solve_radii_lp(c_pert, n, idx_i, idx_j, limits_p)
            if s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_pert.copy()
                best_radii = r_lp.copy()
                
    if best_radii is not None:
        for _ in range(20):
            ok = True
            for i in range(n):
                x, y = best_centers[i]
                r = best_radii[i]
                if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                    ok = False; break
            if ok:
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
                        if d < best_radii[i] + best_radii[j] - 1e-9:
                            ok = False; break
                    if not ok: break
            if ok: break
            best_radii *= 0.99995
            best_sum = np.sum(best_radii)
            
    if best_centers is None:
        best_centers = np.tile(np.linspace(0.1, 0.9, 5), 5).reshape(25, 1)
        best_centers = np.hstack([best_centers, np.repeat(np.linspace(0.1, 0.9, 5), 5).reshape(25, 1)])
        best_centers = np.vstack([best_centers, [[0.5, 0.5]]])
        best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)
