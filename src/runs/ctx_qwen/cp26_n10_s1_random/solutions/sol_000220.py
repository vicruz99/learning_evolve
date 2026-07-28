# sol_000220 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000216 (state 64a1292d) state=2dfecb5a sum of radii=2.390315 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def compute_dists(c):
    """Computes the pairwise Euclidean distance matrix."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def solve_lp_and_grad(c, n, idx_i, idx_j):
    """
    Solves the LP to maximize sum of radii for fixed centers.
    Returns optimal radii, sum of radii, and gradient of sum w.r.t centers.
    """
    dists = compute_dists(c)
    limits = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]),
                        np.minimum(c[:, 1], 1.0 - c[:, 1]))
    bounds = [(0.0, max(lim, 1e-9)) for lim in limits]
    
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    b_ub = dists[idx_i, idx_j]
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success or not np.isfinite(res.fun):
            return None, None, None
    except Exception:
        return None, None, None
        
    r = res.x
    s = -res.fun
    
    grad = np.zeros((n, 2))
    try:
        # Marginals indicate how much the objective improves per unit increase in RHS
        lams = np.abs(res.ineqlin.marginals)
        valid = lams > 1e-7
        if np.any(valid):
            idx_k = np.where(valid)[0]
            lams_v = lams[valid]
            ii = idx_i[idx_k]
            jj = idx_j[idx_k]
            ds = dists[ii, jj]
            ds = np.where(ds > 1e-8, ds, 1.0)
            diffs = c[ii] - c[jj]
            factors = (lams_v[:, None] / ds[:, None])
            
            # Accumulate gradient contributions
            np.add.at(grad, ii, diffs * factors)
            np.add.at(grad, jj, -diffs * factors)
    except Exception:
        pass
    return r, s, grad

def constraint_func(x, n, idx_i, idx_j):
    """Inequality constraints >= 0 for valid packing (used by SLSQP)."""
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    c_mat = np.column_stack((cx, cy))
    d = compute_dists(c_mat)
    return np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r, 
                           d[idx_i, idx_j] - (r[idx_i] + r[idx_j])])

def obj_func(x, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[2*n:])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    idx_i, idx_j = np.triu_indices(n, k=1)
    rng = np.random.default_rng(42)
    
    best_sum = -1.0
    best_c = None
    best_r = None
    
    # Diverse hexagonal row distributions summing to 26
    patterns = [
        [5,6,5,6,4], [6,5,6,5,4], [5,5,6,5,5], [4,6,6,6,4], 
        [6,6,5,5,4], [5,4,6,6,5], [5,6,4,6,5], [6,5,5,6,4],
        [7,5,5,5,4], [4,5,7,5,5], [5,5,5,5,6], [6,6,6,4,4]
    ]
    
    configs = []
    for pat in patterns:
        if sum(pat) != n: continue
        r0 = 0.10
        pts = []
        y = r0
        for ri, cnt in enumerate(pat):
            shift = r0 if ri % 2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) < n: pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
        configs.append(np.array(pts[:n]))
        
    # Add purely random starts to escape lattice biases
    for _ in range(6):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    for cfg in configs:
        c = cfg.copy()
        c += rng.uniform(-0.01, 0.01, c.shape)
        c = np.clip(c, 0.05, 0.95)
        
        curr_r, curr_s, _ = solve_lp_and_grad(c, n, idx_i, idx_j)
        if curr_r is None: continue
        
        step = 0.04
        for it in range(250):
            r, s, g = solve_lp_and_grad(c, n, idx_i, idx_j)
            if r is None: break
            
            gn = np.linalg.norm(g)
            if gn > 1e-6:
                c_new = c + step * g / gn
                c_new = np.clip(c_new, 1e-4, 1.0 - 1e-4)
                r_new, s_new, _ = solve_lp_and_grad(c_new, n, idx_i, idx_j)
                
                if r_new is not None and s_new > s + 1e-7:
                    c = c_new
                    curr_r = r_new
                    curr_s = s_new
                    step = min(step * 1.02, 0.08)
                else:
                    step *= 0.93
            else:
                # Random shake to escape local plateau
                if rng.random() < 0.1:
                    c_shake = c + rng.uniform(-0.02, 0.02, c.shape)
                    c_shake = np.clip(c_shake, 0.05, 0.95)
                    r_sh, s_sh, _ = solve_lp_and_grad(c_shake, n, idx_i, idx_j)
                    if r_sh is not None and s_sh > s:
                        c = c_shake
                        curr_r = r_sh
                        curr_s = s_sh
                        step = 0.04
                        
        if curr_s > best_sum:
            best_sum = curr_s
            best_c = c.copy()
            best_r = curr_r.copy()
            
        # Phase 2: SLSQP Joint Polish to fine-tune centers & radii simultaneously
        x0 = np.concatenate([c.flatten(), curr_r * 0.98])
        bounds_slqp = [(0.0, 1.0)] * (2*n) + [(1e-6, 0.5)] * n
        try:
            res = minimize(obj_func, x0, args=(n,), method='SLSQP', bounds=bounds_slqp,
                           constraints={'type': 'ineq', 'fun': constraint_func, 'args': (n, idx_i, idx_j)},
                           options={'maxiter': 1500, 'ftol': 1e-12})
            if np.isfinite(res.fun):
                c_pol = res.x[:2*n].reshape(n, 2)
                r_pol, s_pol, _ = solve_lp_and_grad(c_pol, n, idx_i, idx_j)
                if r_pol is not None and s_pol > best_sum:
                    best_sum = s_pol
                    best_c = c_pol.copy()
                    best_r = r_pol.copy()
        except Exception:
            pass

    # Phase 3: Strict Safety Scaling to guarantee numerical validity
    if best_c is not None:
        scale = 1.0
        for i in range(n):
            x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
            if r > 1e-12:
                scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
        dists = compute_dists(best_c)
        for i in range(n):
            for j in range(i+1, n):
                d = dists[i, j]
                rs = best_r[i] + best_r[j]
                if rs > 1e-12:
                    scale = min(scale, d / rs)
        best_r *= scale * 0.9999995
        best_sum = float(np.sum(best_r))
    else:
        best_c = rng.uniform(0.1, 0.9, (n, 2))
        best_r = np.full(n, 0.05)
        best_sum = float(np.sum(best_r))
        
    return best_c, best_r, best_sum
