# sol_000345 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000299 (state 3e7613e7) state=e06dc1ba sum of radii=2.603763 correctness=1.0
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

def get_lp_gradient(centers, lp_res):
    """Computes gradient of LP objective w.r.t centers using dual variables."""
    n = centers.shape[0]
    grad = np.zeros((n, 2))
    if lp_res is None:
        return grad
        
    try:
        marg = None
        if hasattr(lp_res, 'marginals'):
            marg = getattr(lp_res.marginals, 'ineqlin', None)
        if marg is None:
            return grad
            
        lams = np.maximum(marg[:M_PAIRS], 0.0)
        mask = lams > 1e-9
        if np.any(mask):
            idx = np.where(mask)[0]
            lam_vals = lams[idx]
            ii = TRIU_I[idx]
            jj = TRIU_J[idx]
            dx = centers[ii, 0] - centers[jj, 0]
            dy = centers[ii, 1] - centers[jj, 1]
            d = np.sqrt(dx**2 + dy**2)
            d = np.where(d < 1e-9, 1e-9, d)
            fx = lam_vals * dx / d
            fy = lam_vals * dy / d
            
            np.add.at(grad[:, 0], ii, fx)
            np.add.at(grad[:, 1], ii, fy)
            np.add.at(grad[:, 0], jj, -fx)
            np.add.at(grad[:, 1], jj, -fy)
    except Exception:
        pass
    return grad

def objective_joint(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def constraint_joint(v):
    """Inequality constraints >= 0 for valid packing."""
    x = v[:N]
    y = v[N:2 * N]
    r = v[2 * N:]
    
    c = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    dx = x[TRIU_I] - x[TRIU_J]
    dy = y[TRIU_I] - y[TRIU_J]
    dr = r[TRIU_I] + r[TRIU_J]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(2025)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # 1. Generate diverse initial configurations
    inits = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [6,6,4,6,4], 
        [5,5,6,5,5], [6,4,6,4,6], [5,6,6,5,4], [6,5,4,6,5],
        [5,5,5,5,6], [6,6,6,4,4], [4,5,6,5,6], [5,4,6,5,6],
        [7,6,5,6,2], [8,6,5,5,2], [9,5,5,5,2]
    ]
    
    for pat in patterns:
        if sum(pat) != N: continue
        pts = []
        r0 = 0.095
        y = r0
        for idx, cnt in enumerate(pat):
            shift = r0 if idx%2 == 1 else 0.0
            x = r0 + shift
            for _ in range(cnt):
                if len(pts) >= N: break
                pts.append([x, y])
                x += 2.0 * r0
            y += r0 * np.sqrt(3)
        pts = np.array(pts[:N])
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = mx - mn + 1e-9
        pts = (pts - mn) / span * 0.72 + 0.14
        inits.append(pts)
        
        for _ in range(3):
            p = pts + rng.uniform(-0.02, 0.02, pts.shape)
            inits.append(np.clip(p, 0.05, 0.95))
            
    for _ in range(6):
        pts = rng.uniform(0.05, 0.95, (N, 2))
        corners = [[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]]
        for i, c in enumerate(corners):
            pts[i] = c + rng.uniform(-0.04, 0.04, 2)
        inits.append(np.clip(pts, 0.02, 0.98))
        
    for _ in range(8):
        inits.append(rng.uniform(0.1, 0.9, (N, 2)))
        
    # 2. Evaluate and optimize each configuration
    for cfg in inits:
        # Phase A: Gradient Ascent with Momentum
        c = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        r, s, lp_res = solve_lp(c)
        if s <= best_sum:
            continue
            
        grad = get_lp_gradient(c, lp_res)
        vel = np.zeros_like(c)
        lr = 0.005
        
        for step in range(800):
            c_new = c + vel
            c_new = np.clip(c_new, 1e-4, 1.0 - 1e-4)
            
            r_new, s_new, lp_res = solve_lp(c_new)
            if s_new > s + 1e-9:
                s = s_new
                c = c_new
                r = r_new
                grad = get_lp_gradient(c, lp_res)
                vel = 0.4 * vel + lr * grad
                lr = min(lr * 1.02, 0.02)
            else:
                lr *= 0.95
                vel *= 0.3
                
        # Phase B: Coordinate Descent with Decaying Step
        curr_c = c.copy()
        curr_s = s
        step_size = 0.015
        
        for _ in range(2500):
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            move = rng.uniform(-step_size, step_size, 2)
            new_pos = np.clip(curr_c[idx] + move, 1e-4, 1.0 - 1e-4)
            curr_c[idx] = new_pos
            
            r_try, s_try, _ = solve_lp(curr_c)
            if s_try > curr_s + 1e-8:
                curr_s = s_try
                if curr_s > best_sum:
                    best_sum = curr_s
                    best_centers = curr_c.copy()
                    best_radii = r_try.copy()
                step_size *= 0.998
            else:
                curr_c[idx] = old
                step_size *= 0.995
                
        # Phase C: Basin Hopping perturbation
        for _ in range(3):
            hop_c = best_centers.copy()
            n_perturb = rng.integers(3, 6)
            idxs = rng.choice(N, n_perturb, replace=False)
            hop_c[idxs] += rng.uniform(-0.03, 0.03, (n_perturb, 2))
            hop_c = np.clip(hop_c, 1e-4, 1.0 - 1e-4)
            
            r_hop, s_hop, _ = solve_lp(hop_c)
            if s_hop > best_sum:
                best_centers = hop_c
                best_radii = r_hop
                best_sum = s_hop
                
    if best_centers is None:
        best_centers = inits[0]
        best_radii, best_sum, _ = solve_lp(best_centers)

    # 3. Final SLSQP Polish
    x0 = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_radii * 0.995])
    bounds_vars = [(0.0, 1.0)] * (2 * N) + [(1e-6, 0.5)] * N
    
    try:
        res = minimize(objective_joint, x0, method='SLSQP', bounds=bounds_vars,
                       constraints={'type': 'ineq', 'fun': constraint_joint},
                       options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
        if np.isfinite(res.fun):
            c_mat = np.column_stack((res.x[:N], res.x[N:2 * N]))
            r_lp, s_lp, _ = solve_lp(c_mat)
            if s_lp > best_sum:
                best_sum = s_lp
                best_centers = c_mat.copy()
                best_radii = r_lp.copy()
    except Exception:
        pass

    # 4. Strict Safety Scaling
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
                
    best_radii *= scale * 0.9999995
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
