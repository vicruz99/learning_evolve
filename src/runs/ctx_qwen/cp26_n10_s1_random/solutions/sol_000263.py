# sol_000263 | problem=circle_packing_26 entrypoint=run_packing
# generation=10 parent=sol_000247 (state 93496474) state=6b909f89 sum of radii=0.133969 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    m = n * (n - 1) // 2
    A_ub = np.zeros((m + n, n))
    b_ub = np.zeros(m + n)
    bounds = [(0.0, None)] * n
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    A_ub[:m, idx_i] = 1.0
    A_ub[:m, idx_j] = 1.0
    b_ub[:m] = np.maximum(dists[idx_i, idx_j], 1e-9)
    
    limits = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                        np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    A_ub[m:, :] = np.eye(n)
    b_ub[m:] = np.maximum(limits, 1e-9)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-9), 0.0

def get_constraints(v, n):
    """Computes inequality constraints for SLSQP: must be >= 0."""
    x = v[:n]
    y = v[n:2*n]
    r = v[2*n:]
    c = []
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dr = r[idx_i] + r[idx_j]
    c.append(dx**2 + dy**2 - dr**2)
    return np.concatenate(c)

def get_objective(v, n):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(v[2*n:])

def constraint_wrapper(v):
    return get_constraints(v, 26)

def objective_wrapper(v):
    return get_objective(v, 26)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    bounds_vars = [(0.01, 0.99)] * (2 * n) + [(1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_wrapper}
    
    # Phase 1: Generate diverse hexagonal and random initial configurations
    starts = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,6,5,5], 
        [5,5,6,5,5], [6,6,5,5,4], [6,5,4,6,5],
        [5,7,4,5,5], [7,5,5,5,4], [5,5,7,5,4]
    ]
    
    for pat in patterns:
        if sum(pat) != n: 
            continue
        pts = []
        r0 = 0.095
        y = r0
        for idx_r, cnt in enumerate(pat):
            shift = r0 if idx_r % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= n: 
                    break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        pts = np.array(pts[:n])
        starts.append(pts)
        
    for _ in range(6):
        starts.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    # Phase 2: Joint SLSQP Optimization from diverse starts
    for cfg in starts:
        c_flat = cfg.flatten()
        r_init = np.full(n, 0.07)
        v0 = np.concatenate([c_flat, r_init])
        
        try:
            res = minimize(objective_wrapper, v0, method='SLSQP', bounds=bounds_vars,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                c_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
                r_lp, s_lp = solve_lp_radii(c_opt)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            pass

    # Phase 3: Coordinate-wise Local Search on Centers
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        
        # Iteratively perturb one circle and re-optimize radii via LP
        for it in range(250):
            idx = rng.integers(n)
            old = curr_c[idx].copy()
            # Adaptive step size decaying over iterations
            step = 0.009 * (0.985 ** (it / 25.0))
            curr_c[idx] += rng.uniform(-step, step, 2)
            curr_c[idx] = np.clip(curr_c[idx], 0.01, 0.99)
            
            r_try, s_try = solve_lp_radii(curr_c)
            if s_try > curr_s + 1e-8:
                curr_s = s_try
                curr_r = r_try.copy()
            else:
                curr_c[idx] = old
                
        best_centers = curr_c
        best_radii = curr_r
        best_sum = curr_s
        
        # Phase 4: Multi-perturbation & SLSQP Polish to escape local traps
        for _ in range(6):
            pert = best_centers + rng.uniform(-0.006, 0.006, best_centers.shape)
            pert = np.clip(pert, 0.02, 0.98)
            r_p, s_p = solve_lp_radii(pert)
            
            if s_p > best_sum - 1e-3:
                v0 = np.concatenate([pert.flatten(), r_p * 0.995])
                try:
                    res = minimize(objective_wrapper, v0, method='SLSQP', bounds=bounds_vars,
                                   constraints=cons, options={'maxiter': 1500, 'ftol': 1e-13})
                    if np.isfinite(res.fun):
                        c_pol = np.column_stack((res.x[:n], res.x[n:2*n]))
                        r_pol, s_pol = solve_lp_radii(c_pol)
                        if s_pol > best_sum:
                            best_centers = c_pol.copy()
                            best_radii = r_pol.copy()
                            best_sum = s_pol
                except Exception:
                    pass

    # Fallback safety net
    if best_centers is None:
        grid = np.array([(i * 0.18 + 0.1, j * 0.18 + 0.1) for j in range(5) for i in range(5)] + [[0.5, 0.5]])
        best_centers = grid[:n]
        best_radii, best_sum = solve_lp_radii(best_centers)

    # Phase 5: Strict Safety Scaling to guarantee numerical validity
    scale = 1.0
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i,0] - best_centers[j,0], 
                         best_centers[i,1] - best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
