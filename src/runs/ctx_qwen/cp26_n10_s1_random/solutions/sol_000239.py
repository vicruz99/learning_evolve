# sol_000239 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000177 (state 0ce77dda) state=fe10e3de sum of radii=2.580604 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    for x, y in centers:
        lim = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(lim, 1e-12)))
        
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    dx = centers[idx_i, 0] - centers[idx_j, 0]
    dy = centers[idx_i, 1] - centers[idx_j, 1]
    b_ub = np.hypot(dx, dy)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return np.full(n, 1e-9)

def force_init(n, seed):
    """Generates a dense initial configuration via iterative force expansion."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.2, 0.8, (n, 2))
    radii = np.full(n, 0.01)
    
    dt = 0.02
    for _ in range(2500):
        radii += 0.00008
        
        forces = np.zeros_like(centers)
        # Boundary repulsion
        for d in range(2):
            viol_left = np.maximum(0.0, radii - centers[:, d])
            viol_right = np.maximum(0.0, centers[:, d] + radii - 1.0)
            forces[:, d] += (viol_right - viol_left) * 500.0
            
        # Pairwise repulsion (vectorized)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dists, 0.0)
        
        required = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = np.maximum(0.0, required - dists)
        
        dir_unit = diff / (dists[:, :, np.newaxis] + 1e-12)
        forces_add = overlap[:, :, np.newaxis] * dir_unit * 100.0
        forces += np.sum(forces_add, axis=1)
        
        centers += forces * dt
        centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
        
    return centers, radii

def joint_obj(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2 * n:])

def joint_cons(x, n):
    """Inequality constraints >= 0 for valid joint packing."""
    cx = x[:n]
    cy = x[n:2 * n]
    r = x[2 * n:]
    
    con = []
    # Boundary constraints
    con.append(cx - r)
    con.append(1.0 - cx - r)
    con.append(cy - r)
    con.append(1.0 - cy - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    cx_m = cx[:, None] - cx[None, :]
    cy_m = cy[:, None] - cy[None, :]
    r_m = r[:, None] + r[None, :]
    idx_i, idx_j = np.triu_indices(n, k=1)
    dist_sq = cx_m[idx_i, idx_j] ** 2 + cy_m[idx_i, idx_j] ** 2
    r_sum_sq = r_m[idx_i, idx_j] ** 2
    con.append(dist_sq - r_sum_sq)
    
    return np.concatenate(con)

def hill_climb(centers, n, rng, steps=3000):
    """Local search on centers using LP evaluation for precise objective."""
    best_centers = centers.copy()
    current_radii = solve_lp_radii(best_centers)
    best_sum = np.sum(current_radii)
    
    step_size = 0.015
    for _ in range(steps):
        i = rng.integers(n)
        old_c = best_centers[i].copy()
        best_centers[i] += rng.uniform(-step_size, step_size, 2)
        best_centers[i] = np.clip(best_centers[i], 1e-4, 1.0 - 1e-4)
        
        new_radii = solve_lp_radii(best_centers)
        new_sum = np.sum(new_radii)
        
        if new_sum > best_sum:
            best_sum = new_sum
            current_radii = new_radii
        else:
            best_centers[i] = old_c
            
        step_size *= 0.9992
        step_size = max(step_size, 1e-5)
        
    return best_centers, solve_lp_radii(best_centers), best_sum

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Force-directed expansion from multiple seeds to find diverse basins
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        centers_init, _ = force_init(n, seed)
        radii_lp = solve_lp_radii(centers_init)
        s_lp = np.sum(radii_lp)
        if s_lp > best_sum:
            best_sum = s_lp
            best_centers = centers_init.copy()
            best_radii = radii_lp.copy()
            
    # Phase 2: Joint SLSQP refinement on centers and radii
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    # Try from best force result and slight perturbations
    candidates = [best_centers]
    candidates.append(np.clip(best_centers + rng.uniform(-0.01, 0.01, (n, 2)), 0.02, 0.98))
    candidates.append(np.clip(best_centers + rng.uniform(-0.02, 0.02, (n, 2)), 0.02, 0.98))
    
    for cfg in candidates:
        r_lp = solve_lp_radii(cfg)
        x0 = np.concatenate([cfg[:, 0], cfg[:, 1], r_lp])
        
        try:
            res = minimize(joint_obj, x0, args=(n,), method='SLSQP',
                           bounds=bounds_vars,
                           constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n,)},
                           options={'maxiter': 6000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2 * n]
                r = np.maximum(res.x[2 * n:], 1e-9)
                centers_opt = np.column_stack((cx, cy))
                radii_opt = r
                s = np.sum(radii_opt)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers_opt.copy()
                    best_radii = radii_opt.copy()
        except Exception:
            pass
            
    # Phase 3: Hill climbing on centers with exact LP evaluation
    best_centers, best_radii, best_sum = hill_climb(best_centers, n, rng, steps=3000)
    
    # Final LP squeeze to guarantee maximal radii for converged centers
    final_radii = solve_lp_radii(best_centers)
    final_sum = np.sum(final_radii)
    if final_sum > best_sum:
        best_radii = final_radii
        best_sum = final_sum
        
    # Phase 4: Strict numerical safety scaling
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
