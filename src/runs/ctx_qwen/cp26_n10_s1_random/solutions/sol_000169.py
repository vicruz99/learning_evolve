# sol_000169 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000160 (state 296f36e1) state=0a5fc00f sum of radii=1.712107 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    bounds = []
    
    # Boundary constraints: r_i <= dist(center_i, boundary)
    for i in range(n):
        x, y = centers[i]
        lims = [x, 1.0 - x, y, 1.0 - y]
        mx = min(lims)
        bounds.append((0.0, max(mx, 1e-9)))
        for lim in lims:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(lim)
            
    # Pairwise non-overlap: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    try:
        res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def obj_sum_r(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2 * n:])

def cons_sum_r(x, n):
    """Inequality constraints >= 0 for valid packing."""
    cx = x[:n]
    cy = x[n:2 * n]
    r = x[2 * n:]
    c = []
    
    # Boundary constraints
    c.extend(cx - r)
    c.extend(1.0 - cx - r)
    c.extend(cy - r)
    c.extend(1.0 - cy - r)
    
    # Pairwise non-overlap constraints (linear distance for better gradients)
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    d = np.sqrt(dx**2 + dy**2 + 1e-12)
    r_sum = r[:, None] + r[None, :]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.extend(d[mask] - r_sum[mask])
    
    return np.array(c)

def generate_initial_configs(n, rng):
    """Generate diverse starting configurations for optimization."""
    configs = []
    # Hexagonal row distributions summing to >= 26
    row_dists = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 6, 6, 5, 4], [6, 6, 5, 5, 4], [5, 5, 5, 6, 5], [5, 7, 5, 5, 4]
    ]
    
    for rd in row_dists:
        pts = []
        y = 0.0
        row = 0
        for cnt in rd:
            x = 0.0
            shift = 0.5 if row % 2 == 1 else 0.0
            for _ in range(cnt):
                if len(pts) >= n:
                    break
                pts.append([x + shift, y])
                x += 1.0
            y += np.sqrt(3) / 2
            row += 1
            
        pts = np.array(pts[:n])
        # Normalize to [0,1] and add margin
        pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0) + 1e-9)
        configs.append(pts * 0.8 + 0.1)
        
        # Perturbed hex
        p = pts * 0.8 + 0.1 + rng.uniform(-0.02, 0.02, (n, 2))
        configs.append(np.clip(p, 0.05, 0.95))
        
    # Grid initialization
    g = np.linspace(0.15, 0.85, 5)
    grid = np.array([[x, y] for y in g for x in g])
    grid = np.vstack([grid, [0.5, 0.5]])
    configs.append(grid[:n])
    
    # Random initializations
    for _ in range(10):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    return configs

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    configs = generate_initial_configs(n, rng)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_opt = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # Phase 1: SLSQP optimization on diverse starts
    for cfg in configs:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2 * n] = cfg[:, 1]
        x0[2 * n:] = 0.04  # Small feasible initial radius
        
        try:
            res = minimize(
                obj_sum_r, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                constraints={'type': 'ineq', 'fun': cons_sum_r, 'args': (n,)},
                options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False}
            )
            if not np.isfinite(res.fun):
                continue
                
            centers_opt = res.x[:2 * n].reshape(n, 2)
            r_lp = solve_lp_radii(centers_opt)
            if r_lp is not None:
                s = np.sum(r_lp)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: LP-guided Hill Climbing on centers
    if best_centers is not None:
        for it in range(50):
            step = 0.025 * (0.96 ** it)
            pert = best_centers + rng.uniform(-step, step, best_centers.shape)
            pert = np.clip(pert, 0.02, 0.98)
            
            r_lp = solve_lp_radii(pert)
            if r_lp is not None:
                s = np.sum(r_lp)
                if s > best_sum:
                    best_sum = s
                    best_centers = pert.copy()
                    best_radii = r_lp.copy()
                    
    # Phase 3: Fine-tune with SLSQP around best configuration
    if best_centers is not None:
        x0 = np.zeros(3 * n)
        x0[:n] = best_centers[:, 0]
        x0[n:2 * n] = best_centers[:, 1]
        x0[2 * n:] = best_radii * 0.98
        
        try:
            res = minimize(
                obj_sum_r, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                constraints={'type': 'ineq', 'fun': cons_sum_r, 'args': (n,)},
                options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False}
            )
            if np.isfinite(res.fun):
                centers_opt = res.x[:2 * n].reshape(n, 2)
                r_lp = solve_lp_radii(centers_opt)
                if r_lp is not None:
                    s = np.sum(r_lp)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers_opt.copy()
                        best_radii = r_lp.copy()
        except Exception:
            pass

    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = configs[0]
        best_radii = np.full(n, 0.08)
        best_sum = np.sum(best_radii)
        
    # Safety scaling to guarantee strict numerical validity against 1e-12 tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
