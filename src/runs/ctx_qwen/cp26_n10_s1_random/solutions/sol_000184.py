# sol_000184 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000118 (state b8add980) state=d59ac4da sum of radii=1.879999 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def objective_eq(v, n):
    """Objective for equal-radius optimization: minimize negative shared radius t."""
    return -v[-1]

def constraints_eq(v, n):
    """Inequality constraints >= 0 for equal-radius packing."""
    cx = v[:n]
    cy = v[n:2*n]
    t = v[2*n]
    
    # Boundary constraints: x >= t, x <= 1-t, y >= t, y <= 1-t
    c = np.concatenate([cx - t, 1.0 - cx - t, cy - t, 1.0 - cy - t])
    
    # Pairwise non-overlap: dist(i,j)^2 >= 4*t^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d2 = dx**2 + dy**2
    idx = np.triu_indices(n, k=1)
    c = np.concatenate([c, d2[idx] - 4.0 * t**2])
    return c

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    lims = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    
    c_obj = -np.ones(n)
    bounds = [(0.0, l) for l in lims]
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, 1e9)
    
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = dists[i, j]
            k += 1
            
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success and np.isfinite(res.fun):
        return res.x, -res.fun
    return np.full(n, 1e-6), 0.0

def penalty_obj(v, n, radii):
    """Smooth penalty function for center optimization with fixed radii."""
    cx = v[:n]
    cy = v[n:2*n]
    pen = 0.0
    
    # Pairwise overlap penalty
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dists = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dists, np.inf)
    r_sum = radii[:, None] + radii[None, :]
    overlap = np.maximum(0, r_sum - dists)
    pen += np.sum(overlap**2)
    
    # Boundary violation penalty
    b_pen_x = np.maximum(0, radii - cx)**2 + np.maximum(0, radii - (1 - cx))**2
    pen += np.sum(b_pen_x)
    b_pen_y = np.maximum(0, radii - cy)**2 + np.maximum(0, radii - (1 - cy))**2
    pen += np.sum(b_pen_y)
    
    return pen

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # 1. Generate diverse hexagonal initial configurations
    configs = []
    patterns = [[5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4]]
    for pat in patterns:
        pts = []
        y = 0.101
        for ri, cnt in enumerate(pat):
            shift = 0.101 if ri % 2 else 0.0
            x = 0.101 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 0.202
            y += 0.101 * np.sqrt(3)
        cfg = np.array(pts[:n])
        cfg = (cfg - cfg.min(axis=0)) / (cfg.max(axis=0) - cfg.min(axis=0)) * 0.88 + 0.06
        configs.append(cfg)
        
    np.random.seed(42)
    for _ in range(6):
        configs.append(np.random.uniform(0.1, 0.9, (n, 2)))

    # 2. Phase 1: SLSQP optimization for equal radii
    bounds_eq = [(0.0, 1.0)] * 2*n + [(0.08, 0.12)]
    for cfg in configs:
        v0 = np.concatenate([cfg[:, 0], cfg[:, 1], [0.095]])
        try:
            res = minimize(objective_eq, v0, args=(n,), method='SLSQP', bounds=bounds_eq,
                           constraints={'type': 'ineq', 'fun': constraints_eq, 'args': (n,)},
                           options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False})
            if -res.fun > 0.101:
                c_opt = res.x[:2*n].reshape(n, 2)
                r_lp, s_lp = solve_lp_radii(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    # 3. Phase 2: Iterative LP + Center Relaxation refinement
    if best_centers is not None:
        for _ in range(6):
            c_x = best_centers[:, 0]
            c_y = best_centers[:, 1]
            x0 = np.concatenate([c_x, c_y])
            
            res_c = minimize(penalty_obj, x0, args=(n, best_radii), method='L-BFGS-B',
                             bounds=[(0.0, 1.0)] * 2*n, options={'maxiter': 1500, 'ftol': 1e-12})
            if res_c.success:
                c_new = res_c.x.reshape(n, 2)
                r_new, s_new = solve_lp_radii(c_new)
                if s_new > best_sum:
                    best_sum = s_new
                    best_centers = c_new.copy()
                    best_radii = r_new.copy()
                    
    # Fallback
    if best_centers is None:
        best_centers = configs[0]
        best_radii, best_sum = solve_lp_radii(best_centers)
        
    # 4. Final safety scaling to strictly satisfy 1e-12 validator tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
