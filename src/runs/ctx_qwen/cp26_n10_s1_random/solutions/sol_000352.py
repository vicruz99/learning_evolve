# sol_000352 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000225 (state c5495767) state=54c047d5 sum of radii=2.625184 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_constraints(params, n, triu_i, triu_j):
    """Computes pairwise non-overlap constraints >= 0."""
    r = params[:n]
    u = params[n:2*n]
    v = params[2*n:3*n]
    
    # Parameterization inherently satisfies boundary constraints
    denom = 1.0 - 2.0 * r
    x = r + denom * u
    y = r + denom * v
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    
    rs = r[:, None] + r[None, :]
    return d2[triu_i, triu_j] - rs[triu_i, triu_j]**2

def objective(params, n):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(params[:n])

def solve_lp_radii(centers, n, triu_i, triu_j):
    """Solves LP to maximize sum of radii for fixed centers."""
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, l) for l in lims]
    
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    dists[np.eye(n, dtype=bool)] = 1e6  # Avoid self-constraints
    
    A = np.zeros((len(triu_i), n))
    A[np.arange(len(triu_i)), triu_i] = 1.0
    A[np.arange(len(triu_i)), triu_j] = 1.0
    b = dists[triu_i, triu_j]
    
    try:
        res = linprog(-np.ones(n), A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def params_to_centers(params, n):
    """Converts parameterization variables back to centers."""
    r = params[:n]
    u = params[n:2*n]
    v = params[2*n:3*n]
    denom = 1.0 - 2.0 * r
    x = r + denom * u
    y = r + denom * v
    return np.column_stack((x, y))

def centers_to_params(centers, n):
    """Converts centers to parameterization variables."""
    r = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    r = np.maximum(r * 0.95, 1e-5)
    denom = np.maximum(1.0 - 2.0 * r, 1e-6)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    triu_i, triu_j = np.triu_indices(n, k=1)
    bounds_v = [(1e-6, 0.5)] * n + [(0.0, 1.0)] * (2 * n)
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n, triu_i, triu_j)}
    
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # 1. Generate diverse initial configurations
    inits = []
    
    # Hexagonal patterns with varying scale and vertical stretch
    for scale in np.linspace(0.85, 1.15, 7):
        for stretch in np.linspace(0.85, 1.15, 5):
            pts = []
            r0 = 0.09 * scale
            dy = np.sqrt(3) * r0 * stretch
            y = r0
            row = 0
            while len(pts) < n:
                shift = r0 if row % 2 == 1 else 0.0
                x = r0 + shift
                while x <= 1.0 - r0 and len(pts) < n:
                    pts.append([x, y])
                    x += 2.0 * r0
                y += dy
                row += 1
            while len(pts) < n:
                pts.append([0.5, 0.5])
            cfg = np.array(pts[:n])
            cfg += rng.uniform(-0.004, 0.004, cfg.shape)
            cfg = np.clip(cfg, 0.01, 0.99)
            inits.append(cfg)
            
    # Dense random starts
    for _ in range(12):
        inits.append(rng.uniform(0.05, 0.95, (n, 2)))
        
    # 2. Phase 1: SLSQP Joint Optimization
    for cfg in inits:
        x0 = centers_to_params(cfg, n)
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds_v,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                cvals = compute_constraints(res.x, n, triu_i, triu_j)
                if np.min(cvals) > -1e-6:
                    centers_opt = params_to_centers(res.x, n)
                    r_lp, s_lp = solve_lp_radii(centers_opt, n, triu_i, triu_j)
                    if r_lp is not None and s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = centers_opt.copy()
                        best_radii = r_lp.copy()
        except Exception:
            pass
            
    # 3. Phase 2: LP-Driven Local Search on Centers
    if best_centers is not None:
        cur_c = best_centers.copy()
        step = 0.02
        
        for _ in range(3000):
            i = rng.integers(n)
            old = cur_c[i].copy()
            
            cur_c[i] = np.clip(cur_c[i] + rng.uniform(-step, step, 2), 1e-4, 0.999)
            
            r_new, s_new = solve_lp_radii(cur_c, n, triu_i, triu_j)
            if r_new is not None and s_new > best_sum + 1e-8:
                best_sum = s_new
                best_centers = cur_c.copy()
                best_radii = r_new.copy()
                step = min(step * 1.015, 0.05)  # Slightly increase on success
                
                # Occasional SLSQP polish to align geometry and radii jointly
                if rng.random() < 0.08:
                    x0 = centers_to_params(cur_c, n)
                    try:
                        res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds_v,
                                       constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13})
                        if np.isfinite(res.fun) and np.min(compute_constraints(res.x, n, triu_i, triu_j)) > -1e-6:
                            c_opt = params_to_centers(res.x, n)
                            r_lp, s_lp = solve_lp_radii(c_opt, n, triu_i, triu_j)
                            if r_lp is not None and s_lp > best_sum:
                                best_sum = s_lp
                                best_centers = c_opt.copy()
                                best_radii = r_lp.copy()
                                cur_c = best_centers.copy()
                    except Exception:
                        pass
            else:
                cur_c[i] = old
                step *= 0.997  # Gradual decay on failure

    # 4. Phase 3: Final SLSQP Polish
    x0 = centers_to_params(best_centers, n)
    try:
        res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds_v,
                       constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14})
        if np.isfinite(res.fun) and np.min(compute_constraints(res.x, n, triu_i, triu_j)) > -1e-6:
            c_opt = params_to_centers(res.x, n)
            r_lp, s_lp = solve_lp_radii(c_opt, n, triu_i, triu_j)
            if r_lp is not None and s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_opt.copy()
                best_radii = r_lp.copy()
    except Exception:
        pass

    # 5. Strict Safety Scaling for Validator Tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    d = np.sqrt((best_centers[triu_i, 0] - best_centers[triu_j, 0])**2 + 
                (best_centers[triu_i, 1] - best_centers[triu_j, 1])**2)
    rs = best_radii[triu_i] + best_radii[triu_j]
    mask = rs > 1e-12
    if np.any(mask):
        scale = min(scale, np.min(d[mask] / rs[mask]))
        
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
