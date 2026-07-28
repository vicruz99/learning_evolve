# sol_000316 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000276 (state cc798eee) state=3acc8ded sum of radii=1.979587 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
IDX_I, IDX_J = np.triu_indices(N, k=1)

def solve_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    m = len(IDX_I)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), IDX_I] = 1.0
    A_ub[np.arange(m), IDX_J] = 1.0
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[IDX_I, IDX_J]
    
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, l) for l in lims]
    
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success and np.isfinite(res.fun):
        return res.x, -res.fun, res
    return None, 0.0, None

def lp_objective_and_grad(x):
    """Objective and gradient for L-BFGS-B: maximize LP sum of radii."""
    c = x.reshape(N, 2)
    r, s, res = solve_lp(c)
    if r is None:
        return 1e6, np.zeros_like(x)
    
    grad = np.zeros_like(c)
    try:
        marg = getattr(getattr(res, 'marginals', None), 'ineqlin', None)
        if marg is None:
            return -s, grad.flatten()
        marg = np.asarray(marg)
        valid = marg > 1e-7
        if not np.any(valid):
            return -s, grad.flatten()
        
        k = np.where(valid)[0]
        i_idx = IDX_I[k]
        j_idx = IDX_J[k]
        lams = marg[k]
        
        diff = c[i_idx] - c[j_idx]
        dist = np.linalg.norm(diff, axis=1)
        dist_safe = np.maximum(dist, 1e-12)
        factors = (lams / dist_safe)[:, np.newaxis]
        
        np.add.at(grad, i_idx, diff * factors)
        np.add.at(grad, j_idx, -diff * factors)
    except Exception:
        pass
    return -s, -grad.flatten()

def slsqp_objective(x):
    """Objective for joint SLSQP: minimize negative sum of radii."""
    return -np.sum(x[2*N:])

def slsqp_constraints(x):
    """Inequality constraints >= 0 for SLSQP joint optimization."""
    cx = x[:N]
    cy = x[N:2*N]
    r = x[2*N:]
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    d2 = dx**2 + dy**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    pc = d2[IDX_I, IDX_J] - rs[IDX_I, IDX_J]**2
    return np.concatenate([c, pc])

def generate_hex_config(row_counts, r0, rng):
    """Generates a hexagonal lattice configuration based on row counts."""
    pts = []
    y = r0
    for idx, cnt in enumerate(row_counts):
        shift = r0 if idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= N:
                break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
    while len(pts) < N:
        pts.append([0.5, 0.5])
    return np.array(pts[:N])

def jiggle_centers(centers, radii, rng):
    """Local coordinate descent to escape local minima."""
    curr_c = centers.copy()
    curr_r = radii.copy()
    curr_s = np.sum(curr_r)
    step = 0.012
    for _ in range(600):
        i = rng.integers(N)
        old = curr_c[i].copy()
        curr_c[i] += rng.uniform(-step, step, 2)
        curr_c[i] = np.clip(curr_c[i], 1e-4, 1.0 - 1e-4)
        r_try, s_try, _ = solve_lp(curr_c)
        if r_try is not None and s_try > curr_s + 1e-8:
            curr_s = s_try
            curr_r = r_try.copy()
            step *= 0.995
        else:
            curr_c[i] = old
            if rng.random() < 0.05:
                step *= 0.9
    return curr_c, curr_r, curr_s

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Diverse hexagonal row patterns summing to 26
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,5,5], 
        [6,6,5,5,4], [5,7,5,5,4], [4,5,6,6,5], [5,4,6,6,5]
    ]
    
    configs = []
    for pat in patterns:
        if sum(pat) < N: continue
        cfg = generate_hex_config(pat, 0.095, rng)
        # Normalize to fit comfortably inside [0.06, 0.94]
        mn = cfg.min(axis=0)
        mx = cfg.max(axis=0)
        span = mx - mn + 1e-9
        cfg_norm = (cfg - mn) / span * 0.88 + 0.06
        configs.append(cfg_norm)
        
        for _ in range(2):
            p = cfg_norm + rng.uniform(-0.02, 0.02, cfg_norm.shape)
            configs.append(np.clip(p, 0.04, 0.96))
            
    for _ in range(5):
        configs.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    bounds_slqp = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
    
    # Phase 1: Joint SLSQP optimization from multiple starts
    for cfg in configs:
        r_init, _, _ = solve_lp(cfg)
        if r_init is None:
            r_init = np.full(N, 0.08)
            
        x0 = np.concatenate([cfg.flatten(), r_init])
        
        try:
            res = minimize(slsqp_objective, x0, method='SLSQP', bounds=bounds_slqp,
                           constraints={'type': 'ineq', 'fun': slsqp_constraints},
                           options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            if np.isfinite(res.fun):
                c_opt = res.x[:2*N].reshape(N, 2)
                r_opt, s_opt, _ = solve_lp(c_opt)
                if r_opt is not None and s_opt > best_sum:
                    best_sum = s_opt
                    best_centers = c_opt.copy()
                    best_radii = r_opt.copy()
        except Exception:
            pass
            
    if best_centers is not None:
        # Phase 2: LP Dual Gradient Ascent (L-BFGS-B)
        curr_c = best_centers.copy()
        for _ in range(3):
            try:
                res_lp = minimize(lp_objective_and_grad, curr_c.flatten(), method='L-BFGS-B', jac=True,
                                  bounds=[(0.001, 0.999)] * (2 * N),
                                  options={'maxiter': 1500, 'ftol': 1e-14})
                if np.isfinite(res_lp.fun):
                    curr_c = res_lp.x.reshape(N, 2)
                    r_lp, s_lp, _ = solve_lp(curr_c)
                    if r_lp is not None and s_lp > best_sum:
                        best_sum = s_lp
                        best_centers = curr_c.copy()
                        best_radii = r_lp.copy()
            except Exception:
                pass
                
        # Phase 3: Jiggle Search
        curr_c, curr_r, curr_s = jiggle_centers(best_centers, best_radii, rng)
        if curr_s > best_sum:
            best_sum = curr_s
            best_centers = curr_c.copy()
            best_radii = curr_r.copy()
            
        # Phase 4: Random restarts around best configuration
        for _ in range(6):
            x_pert = best_centers.flatten() + rng.uniform(-0.015, 0.015, 2*N)
            x_pert = np.clip(x_pert, 0.01, 0.99)
            try:
                res_p = minimize(lp_objective_and_grad, x_pert, method='L-BFGS-B', jac=True,
                                 bounds=[(0.001, 0.999)] * (2 * N),
                                 options={'maxiter': 800, 'ftol': 1e-13})
                if np.isfinite(res_p.fun):
                    c_p = res_p.x.reshape(N, 2)
                    r_p, s_p, _ = solve_lp(c_p)
                    if r_p is not None and s_p > best_sum:
                        best_sum = s_p
                        best_centers = c_p.copy()
                        best_radii = r_p.copy()
            except Exception:
                pass
                
    # Fallback if optimization fails completely
    if best_centers is None:
        best_centers = generate_hex_config([6,5,6,5,4], 0.09, rng)
        best_radii, best_sum, _ = solve_lp(best_centers)
        
    # Final strict safety scaling to guarantee numerical validity against 1e-12 tolerance
    scale = 1.0
    for i in range(N):
        x, y = best_centers[i]
        r = best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(best_centers[i,0] - best_centers[j,0], 
                         best_centers[i,1] - best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    r_final = best_radii * scale * 0.9999995
    s_final = float(np.sum(r_final))
    
    return best_centers, r_final, s_final
