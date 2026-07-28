# sol_000343 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000299 (state 3e7613e7) state=dee79e77 sum of radii=2.298220 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
TRIU_I, TRIU_J = np.triu_indices(N, k=1)
M_PAIRS = len(TRIU_I)

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, l) for l in lims]
    
    c_obj = -np.ones(n)
    A_ub = np.zeros((M_PAIRS, n))
    A_ub[np.arange(M_PAIRS), TRIU_I] = 1.0
    A_ub[np.arange(M_PAIRS), TRIU_J] = 1.0
    
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    b_ub = dists[TRIU_I, TRIU_J]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun, res
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0, None

def lp_obj_grad(centers_flat):
    """Objective and gradient for center optimization via LP duals."""
    centers = centers_flat.reshape(N, 2)
    r, s, res = solve_lp(centers)
    if r is None:
        return 1e6, np.zeros_like(centers_flat)
        
    grad = np.zeros_like(centers)
    if res is not None:
        try:
            marginals = None
            if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
                marginals = np.asarray(res.marginals.ineqlin)
            elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
                marginals = np.asarray(res.ineqlin.marginals)
                
            if marginals is not None:
                gains = np.maximum(marginals[:M_PAIRS], 0.0)
                if np.any(gains > 1e-8):
                    valid = gains > 1e-8
                    ii = TRIU_I[valid]
                    jj = TRIU_J[valid]
                    g = gains[valid]
                    diff = centers[ii] - centers[jj]
                    d_ij = np.linalg.norm(diff, axis=1)
                    d_safe = np.where(d_ij > 1e-8, d_ij, 1.0)
                    vecs = (g / d_safe)[:, np.newaxis] * diff
                    np.add.at(grad, ii, vecs)
                    np.add.at(grad, jj, -vecs)
        except Exception:
            pass
    return -s, -grad.flatten()

def equal_radius_obj(v):
    """Objective for equal-radius packing: minimize -R"""
    return -v[-1]

def equal_radius_cons(v):
    """Inequality constraints >= 0 for equal-radius packing"""
    cx = v[:N]
    cy = v[N:2*N]
    r = v[2*N]
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    dx = cx[TRIU_I] - cx[TRIU_J]
    dy = cy[TRIU_I] - cy[TRIU_J]
    c = np.concatenate([c, dx**2 + dy**2 - 4.0*r**2])
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # 1. Generate diverse initial configurations
    inits = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [4, 6, 5, 6, 5], [6, 6, 4, 6, 4], 
        [5, 5, 6, 5, 5], [6, 4, 6, 4, 6], [5, 6, 6, 5, 4], [6, 5, 4, 6, 5],
        [5, 5, 5, 5, 6], [6, 6, 6, 4, 4], [4, 5, 6, 5, 6], [5, 4, 6, 5, 6],
        [7, 6, 5, 6, 2], [8, 6, 5, 5, 2], [9, 5, 5, 5, 2],
        [5, 7, 5, 6, 3], [6, 7, 5, 5, 3], [7, 7, 6, 6, 0]
    ]
    
    for pat in patterns:
        if sum(pat) != N: continue
        pts = []
        r0 = 0.10
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= N: break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        pts = np.array(pts[:N])
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = mx - mn + 1e-9
        pts = (pts - mn) / span * 0.88 + 0.06
        inits.append(pts)
        
        for _ in range(4):
            p = pts + rng.uniform(-0.025, 0.025, pts.shape)
            inits.append(np.clip(p, 0.02, 0.98))

    for _ in range(12):
        pts = rng.uniform(0.1, 0.9, (N, 2))
        inits.append(pts)
        
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    
    # Phase 1: L-BFGS-B optimization on centers using LP dual gradients
    for cfg in inits:
        c0 = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(lp_obj_grad, c0.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_c,
                           options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp(c_opt)
                if r_opt is not None and s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            continue

    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum, _ = solve_lp(best_centers)

    # Phase 2: Equal-radius SLSQP warm start to find symmetric basins
    bounds_eq = [(0.0, 1.0)] * (2 * N) + [(0.04, 0.15)]
    cons_eq = {'type': 'ineq', 'fun': equal_radius_cons}
    
    for cfg in inits[:10]:
        v0 = np.concatenate([cfg.flatten(), [0.09]])
        try:
            res_eq = minimize(equal_radius_obj, v0, method='SLSQP', bounds=bounds_eq,
                              constraints=cons_eq, options={'maxiter': 4000, 'ftol': 1e-13})
            if np.isfinite(res_eq.fun):
                c_eq = res_eq.x[:2*N].reshape(N, 2)
                r_lp, s_lp, _ = solve_lp(c_eq)
                if r_lp is not None and s_lp > best_sum:
                    best_sum = s_lp
                    best_centers = c_eq.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue

    # Phase 3: Adaptive Local Coordinate Search
    curr_c = best_centers.copy()
    curr_r, curr_s, _ = solve_lp(curr_c)
    step = 0.012
    
    for it in range(4000):
        idx = rng.integers(N)
        old = curr_c[idx].copy()
        move = rng.uniform(-step, step, 2)
        new_c = np.clip(curr_c[idx] + move, 1e-4, 1.0 - 1e-4)
        curr_c[idx] = new_c
        
        r_try, s_try, _ = solve_lp(curr_c)
        if r_try is not None and s_try > curr_s + 1e-9:
            curr_s = s_try
            curr_r = r_try.copy()
            if curr_s > best_sum:
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
            step = min(step * 1.003, 0.04)
        else:
            curr_c[idx] = old
            step *= 0.996
        if step < 1e-5: break

    # Phase 4: Basin-hopping style random restarts from best
    for _ in range(25):
        pert = best_centers + rng.uniform(-0.007, 0.007, best_centers.shape)
        pert = np.clip(pert, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(lp_obj_grad, pert.flatten(), method='L-BFGS-B', jac=True, bounds=bounds_c,
                           options={'maxiter': 3000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp(c_opt)
                if r_opt is not None and s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            pass

    # Phase 5: Strict Safety Scaling to guarantee numerical validity
    scale = 1.0
    for i in range(N):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999998
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
