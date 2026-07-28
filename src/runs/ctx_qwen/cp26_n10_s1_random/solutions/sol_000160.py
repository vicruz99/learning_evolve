# sol_000160 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000126 (state 8609ace4) state=296f36e1 sum of radii=2.624822 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def obj_sum_r(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2*n:])

def cons_sum_r(x, n):
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
    
    # Pairwise non-overlap constraints: squared distance >= squared sum of radii
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    d2 = cx_m**2 + cy_m**2
    rs2 = r_m**2
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.extend(d2[mask] - rs2[mask])
    return np.array(c)

def generate_initial_configs(n, rng):
    """Generate diverse starting configurations for optimization."""
    configs = []
    # Hexagonal lattices with varying base radii
    for r0 in np.linspace(0.085, 0.105, 6):
        pts = []
        y = r0
        row = 0
        while y + r0 < 1.0 and len(pts) < n:
            shift = r0 if row % 2 == 1 else 0.0
            x = r0 + shift
            while x + r0 < 1.0 and len(pts) < n:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        while len(pts) < n:
            pts.append([0.5, 0.5])
        configs.append(np.array(pts[:n]))
        # Add perturbed versions to break symmetry
        pert = np.array(pts[:n]) + rng.uniform(-0.015, 0.015, (n, 2))
        configs.append(np.clip(pert, 0.05, 0.95))
        
    # Grid-based initialization
    grid = np.array([[0.1 + i*0.18, 0.1 + j*0.18] for j in range(5) for i in range(5)])
    grid = np.vstack([grid, [0.5, 0.5]])
    configs.append(grid)
    configs.append(grid + rng.uniform(-0.02, 0.02, (n, 2)))
    return configs

def solve_lp_radii(centers):
    """Solve LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    bounds = []
    
    for i in range(n):
        x, y = centers[i]
        mx = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(mx, 1e-9)))
        for lim in [x, 1.0 - x, y, 1.0 - y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(lim)
            
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

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    configs = generate_initial_configs(n, rng)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_opt = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.5)] * n
    
    # Phase 1: Initial SLSQP optimization on diverse starts
    for cfg in configs:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2 * n] = cfg[:, 1]
        x0[2 * n:] = 0.075  # Strictly feasible initial radius
        
        try:
            res = minimize(
                obj_sum_r, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                constraints={'type': 'ineq', 'fun': cons_sum_r, 'args': (n,)},
                options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False}
            )
            if not np.isfinite(res.fun):
                continue
                
            cx = res.x[:n]
            cy = res.x[n:2 * n]
            centers_opt = np.column_stack((cx, cy))
            
            # Refine radii with LP for fixed centers
            r_lp = solve_lp_radii(centers_opt)
            if r_lp is not None:
                s = np.sum(r_lp)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Perturbation refinement on best result to escape local minima
    if best_centers is not None:
        for _ in range(8):
            pert = best_centers + rng.normal(0, 0.005, best_centers.shape)
            pert = np.clip(pert, 0.01, 0.99)
            
            x0 = np.zeros(3 * n)
            x0[:n] = pert[:, 0]
            x0[n:2 * n] = pert[:, 1]
            x0[2 * n:] = np.mean(best_radii) * 0.95
            
            try:
                res = minimize(
                    obj_sum_r, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                    constraints={'type': 'ineq', 'fun': cons_sum_r, 'args': (n,)},
                    options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False}
                )
                if not np.isfinite(res.fun):
                    continue
                    
                cx = res.x[:n]
                cy = res.x[n:2 * n]
                centers_opt = np.column_stack((cx, cy))
                
                r_lp = solve_lp_radii(centers_opt)
                if r_lp is not None:
                    s = np.sum(r_lp)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers_opt.copy()
                        best_radii = r_lp.copy()
            except Exception:
                continue

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
