# sol_000125 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000081 (state 6da8454c) state=c1034f91 sum of radii=2.384629 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def get_joint_constraints(vars_array, n):
    """Computes inequality constraints >= 0 for valid joint packing."""
    xs = vars_array[0::3]
    ys = vars_array[1::3]
    rs = vars_array[2::3]
    
    con = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    
    xs_m = xs[:, None] - xs[None, :]
    ys_m = ys[:, None] - ys[None, :]
    rs_m = rs[:, None] + rs[None, :]
    
    d2 = xs_m**2 + ys_m**2
    r2 = rs_m**2
    
    idx = np.triu_indices(n, k=1)
    con = np.concatenate([con, d2[idx] - r2[idx]])
    return con

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    # Boundary upper bounds
    ub = np.min(np.minimum(centers, 1.0 - centers), axis=1)
    ub = np.maximum(ub, 1e-9)
    
    # Pairwise distances
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # LP constraints: r_i + r_j <= dist_ij
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
            
    c = -np.ones(n)
    bounds = [(0.0, ub[i]) for i in range(n)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return -res.fun, res.x
    except Exception:
        pass
        
    # Fallback to equal radii if LP fails
    r_fall = np.min(ub) * 0.99
    return n * r_fall, np.full(n, r_fall)

def obj_lp(centers_flat):
    """Objective for center optimization: negative sum of LP radii."""
    centers = centers_flat.reshape(-1, 2)
    s, _ = solve_lp_radii(centers)
    return -s

def generate_initial_configs(n):
    """Generates diverse hexagonal and perturbed initial configurations."""
    inits = []
    row_configs = [
        [6, 5, 5, 5, 5], [5, 6, 5, 5, 5], [5, 5, 6, 5, 5],
        [5, 5, 5, 6, 5], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4],
        [5, 5, 5, 5, 5, 1], [4, 6, 6, 6, 4], [5, 6, 6, 5, 4],
        [6, 5, 5, 5, 5]
    ]
    rng = np.random.default_rng(42)
    
    for rc in row_configs:
        pts = []
        y = 0.1
        r0 = 0.095
        for idx, cnt in enumerate(rc):
            shift = r0 if idx % 2 == 1 else 0.0
            row_w = (cnt - 1) * 2 * r0
            x_start = 0.5 - row_w / 2.0 + shift
            for c in range(cnt):
                if len(pts) >= n: break
                pts.append([x_start + c * 2 * r0, y])
            y += np.sqrt(3) * r0
        inits.append(np.array(pts[:n]))
        
    # Add controlled perturbations
    for _ in range(6):
        base = inits[0].copy()
        base += rng.uniform(-0.025, 0.025, base.shape)
        inits.append(np.clip(base, 0.05, 0.95))
        
    return inits

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -1.0
    best_centers = None
    bounds_c = [(0.0, 1.0)] * (2 * n)
    inits = generate_initial_configs(n)
    
    # Phase 1: Derivative-free center optimization using LP objective
    for cfg in inits:
        try:
            res = minimize(obj_lp, cfg.flatten(), method='Powell',
                           bounds=bounds_c, options={'maxiter': 400, 'fatol': 1e-12})
            if np.isfinite(res.fun):
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_centers = res.x.reshape(-1, 2)
        except Exception:
            continue
            
    # Phase 2: Joint SLSQP refinement to polish both centers and radii
    if best_centers is not None:
        _, r_init = solve_lp_radii(best_centers)
        x0 = np.zeros(3 * n)
        x0[0::3] = best_centers[:, 0]
        x0[1::3] = best_centers[:, 1]
        x0[2::3] = r_init
        
        def obj_joint(v):
            return -np.sum(v[2::3])
            
        cons = {'type': 'ineq', 'fun': get_joint_constraints, 'args': (n,)}
        bounds_joint = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
        
        try:
            res2 = minimize(obj_joint, x0, method='SLSQP', bounds=bounds_joint,
                            constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13})
            if np.isfinite(res2.fun):
                cx = res2.x[0::3]
                cy = res2.x[1::3]
                r = res2.x[2::3]
                c_tmp = np.column_stack((cx, cy))
                
                # Quick validity check
                valid = True
                for i in range(n):
                    if cx[i] < r[i] - 1e-12 or cx[i] > 1 - r[i] + 1e-12 or \
                       cy[i] < r[i] - 1e-12 or cy[i] > 1 - r[i] + 1e-12:
                        valid = False; break
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            d2 = (cx[i] - cx[j])**2 + (cy[i] - cy[j])**2
                            rs = r[i] + r[j]
                            if d2 < rs**2 - 1e-12:
                                valid = False; break
                        if not valid: break
                if valid:
                    s = np.sum(r)
                    if s > best_sum:
                        best_sum = s
                        best_centers = c_tmp
        except Exception:
            pass
            
    # Phase 3: Final exact LP radius assignment on optimized centers
    if best_centers is None:
        best_centers = inits[0]
        
    final_sum, final_radii = solve_lp_radii(best_centers)
    
    # Safety scaling to strictly satisfy validator tolerance
    final_radii *= 0.9999995
    best_sum = float(np.sum(final_radii))
    
    # Final clip to avoid edge-case numerical artifacts
    best_centers = np.clip(best_centers, 1e-9, 1.0 - 1e-9)
    
    return best_centers, final_radii, best_sum
