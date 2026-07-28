# sol_000170 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000160 (state 296f36e1) state=1dbaa47e sum of radii=2.616537 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def obj_func(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2*n:])

def cons_func(x, n):
    """Inequality constraints >= 0 for valid packing."""
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    c = []
    # Boundary constraints
    c.extend(cx - r)
    c.extend(1.0 - cx - r)
    c.extend(cy - r)
    c.extend(1.0 - cy - r)
    
    # Pairwise non-overlap constraints (squared for smooth gradients)
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    idx = np.triu_indices(n, k=1)
    d2 = cx_m[idx]**2 + cy_m[idx]**2
    rs2 = r_m[idx]**2
    c.extend(d2 - rs2)
    return np.array(c)

def solve_lp_radii(centers):
    """Solve LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    bounds = []
    
    for i in range(n):
        x, y = centers[i]
        lim = min(x, 1.0-x, y, 1.0-y)
        bounds.append((0.0, max(lim, 1e-9)))
        for val in [x, 1.0-x, y, 1.0-y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(val)
            
    for i in range(n):
        for j in range(i+1, n):
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

def generate_initial_configs(n, rng):
    """Generate diverse starting configurations."""
    configs = []
    
    # Hexagonal lattices with varying row distributions
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 5, 5, 6], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5]
    ]
    for pat in patterns:
        pts = []
        r0 = 0.09
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) < n:
                    pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
        configs.append(np.array(pts[:n]))
        
    # Grid-based
    g = np.linspace(0.12, 0.88, 5)
    grid = np.array([[x,y] for y in g for x in g])
    grid = np.vstack([grid, [0.5, 0.5]])
    configs.append(grid)
    
    # Diagonal/Corner focused
    diag_pts = []
    for i in range(6):
        diag_pts.append([0.05 + i*0.15, 0.05 + i*0.15])
        diag_pts.append([0.95 - i*0.15, 0.95 - i*0.15])
    while len(diag_pts) < n:
        diag_pts.append([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)])
    configs.append(np.array(diag_pts[:n]))
    
    # Random dense
    for _ in range(20):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    return configs

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    configs = generate_initial_configs(n, rng)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_opt = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.5)] * n
    
    # Phase 1: Multi-start SLSQP + LP refinement
    for cfg in configs:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2*n] = cfg[:, 1]
        x0[2*n:] = 0.075  # Feasible initial radius
        
        try:
            res = minimize(
                obj_func, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                constraints={'type': 'ineq', 'fun': cons_func, 'args': (n,)},
                options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False}
            )
            if not np.isfinite(res.fun):
                continue
                
            cx = res.x[:n]
            cy = res.x[n:2*n]
            centers_opt = np.column_stack((cx, cy))
            
            # LP refinement for exact radii given centers
            r_lp = solve_lp_radii(centers_opt)
            if r_lp is not None:
                s = np.sum(r_lp)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Perturbation refinement on best result
    if best_centers is not None:
        for scale in [0.3, 0.6, 1.0, 1.5, 2.0]:
            for _ in range(3):
                pert = best_centers + rng.normal(0, 0.004 * scale, best_centers.shape)
                pert = np.clip(pert, 0.02, 0.98)
                
                x0 = np.zeros(3 * n)
                x0[:n] = pert[:, 0]
                x0[n:2*n] = pert[:, 1]
                x0[2*n:] = np.mean(best_radii) * 0.95
                
                try:
                    res = minimize(
                        obj_func, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                        constraints={'type': 'ineq', 'fun': cons_func, 'args': (n,)},
                        options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False}
                    )
                    if not np.isfinite(res.fun):
                        continue
                        
                    centers_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
                    r_lp = solve_lp_radii(centers_opt)
                    if r_lp is not None:
                        s = np.sum(r_lp)
                        if s > best_sum:
                            best_sum = s
                            best_centers = centers_opt.copy()
                            best_radii = r_lp.copy()
                except Exception:
                    continue

    # Fallback
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
