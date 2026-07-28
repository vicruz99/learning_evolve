# sol_000171 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000160 (state 296f36e1) state=d3cd6227 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """
    Solves the LP to maximize sum of radii for fixed centers.
    Constraints: r_i + r_j <= dist(i,j) and 0 <= r_i <= dist(i, boundary)
    """
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    # Boundary limits for each circle
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]), 
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9) # Ensure strictly positive upper bounds
    bounds = [(0.0, lim) for lim in lims]
    
    # Pairwise distance constraints: r_i + r_j <= d_ij
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    idx = np.triu_indices(n, k=1)
    m = n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = dists[idx]
    
    for k, (i, j) in enumerate(zip(idx[0], idx[1])):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return np.maximum(res.x, 0.0)
    except Exception:
        pass
    return None

def obj_slsqp(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2*n:])

def cons_slsqp(x, n):
    """Inequality constraints >= 0 for valid packing."""
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    # Boundary constraints
    c = np.concatenate([
        cx - r, 1.0 - cx - r,
        cy - r, 1.0 - cy - r
    ])
    
    # Pairwise non-overlap: d^2 >= (r_i + r_j)^2
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    d2 = cx_m**2 + cy_m**2
    rs2 = r_m**2
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c = np.concatenate([c, d2[mask] - rs2[mask]])
    return c

def generate_initial_configs(n, rng):
    """Generate diverse starting configurations."""
    configs = []
    
    # 1. Hexagonal lattices with varying densities
    for r0 in np.linspace(0.07, 0.11, 5):
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
        # Perturbed hex
        pert = np.array(pts[:n]) + rng.uniform(-0.012, 0.012, (n, 2))
        configs.append(np.clip(pert, 0.02, 0.98))
        
    # 2. Grid patterns
    for sp in [0.18, 0.20]:
        grid = np.array([[0.5 - 2*sp + i*sp, 0.5 - 2*sp + j*sp] 
                         for j in range(5) for i in range(5)])
        grid = np.vstack([grid, [0.5, 0.5]])
        configs.append(grid[:n])
        
    # 3. Random dense placements
    for _ in range(10):
        cfg = rng.uniform(0.1, 0.9, (n, 2))
        configs.append(cfg)
        
    return configs

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    configs = generate_initial_configs(n, rng)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_opt = [(0.0, 1.0)] * (2 * n) + [(1e-7, 0.5)] * n
    
    # Phase 1: SLSQP Multi-Start
    for cfg in configs:
        x0 = np.zeros(3 * n)
        x0[:n] = cfg[:, 0]
        x0[n:2*n] = cfg[:, 1]
        x0[2*n:] = 0.075
        
        try:
            res = minimize(
                obj_slsqp, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                constraints={'type': 'ineq', 'fun': cons_slsqp, 'args': (n,)},
                options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False}
            )
            if not np.isfinite(res.fun):
                continue
                
            cx = res.x[:n]
            cy = res.x[n:2*n]
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

    # Phase 2: LP-Driven Local Search on Centers
    if best_centers is not None:
        centers = best_centers.copy()
        step = 0.015
        for _ in range(800):
            # Pick random circle to perturb
            i = rng.integers(n)
            old_pos = centers[i].copy()
            
            # Perturb
            centers[i] += rng.uniform(-step, step, 2)
            centers[i] = np.clip(centers[i], 1e-4, 1.0 - 1e-4)
            
            r_lp = solve_lp_radii(centers)
            if r_lp is not None:
                curr_sum = np.sum(r_lp)
                if curr_sum > best_sum + 1e-8:
                    best_sum = curr_sum
                    best_centers = centers.copy()
                    best_radii = r_lp.copy()
                    step *= 0.998 # Slow decay on success
                else:
                    centers[i] = old_pos # Revert
                    if rng.random() < 0.05:
                        step *= 0.95 # Faster decay occasionally
            else:
                centers[i] = old_pos
                
        # Phase 3: Final SLSQP polish from refined centers
        x0 = np.zeros(3 * n)
        x0[:n] = best_centers[:, 0]
        x0[n:2*n] = best_centers[:, 1]
        x0[2*n:] = np.mean(best_radii) * 0.98
        
        try:
            res = minimize(
                obj_slsqp, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                constraints={'type': 'ineq', 'fun': cons_slsqp, 'args': (n,)},
                options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False}
            )
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2*n]
                centers_opt = np.column_stack((cx, cy))
                r_lp = solve_lp_radii(centers_opt)
                if r_lp is not None:
                    s = np.sum(r_lp)
                    if s > best_sum:
                        best_sum = s
                        best_centers = centers_opt.copy()
                        best_radii = r_lp.copy()
        except Exception:
            pass

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
