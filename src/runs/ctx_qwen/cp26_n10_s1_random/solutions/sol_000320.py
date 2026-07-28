# sol_000320 | problem=circle_packing_26 entrypoint=run_packing
# generation=14 parent=sol_000276 (state cc798eee) state=210aafc7 sum of radii=2.581008 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def solve_lp_radii(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    # Boundary limits
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    bounds = [(0.0, l) for l in lims]
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    diffs = centers[idx_i] - centers[idx_j]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    b_ub = dists
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun, res
    except Exception:
        pass
    return np.full(n, 1e-6), 1e-6, None

def compute_lp_gradient(centers, res):
    """Computes gradient of LP sum of radii w.r.t centers using dual variables."""
    if res is None:
        return np.zeros_like(centers)
    n = centers.shape[0]
    grad = np.zeros_like(centers)
    try:
        # Extract dual multipliers for pairwise constraints
        marg = None
        if hasattr(res, 'marginals') and hasattr(res.marginals, 'ineqlin'):
            marg = np.asarray(res.marginals.ineqlin)
        elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
            marg = np.asarray(res.ineqlin.marginals)
            
        if marg is None or len(marg) == 0:
            return grad
            
        idx_i, idx_j = np.triu_indices(n, k=1)
        lams = marg
        mask = lams > 1e-8
        k = np.where(mask)[0]
        if len(k) == 0:
            return grad
            
        ii = idx_i[k]
        jj = idx_j[k]
        lam = lams[k]
        diff = centers[ii] - centers[jj]
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist = np.where(dist < 1e-12, 1e-12, dist)
        factors = (lam / dist)
        
        # Gradient pushes active pairs apart
        np.add.at(grad[:, 0], ii, factors * diff[:, 0])
        np.add.at(grad[:, 1], ii, factors * diff[:, 1])
        np.add.at(grad[:, 0], jj, -factors * diff[:, 0])
        np.add.at(grad[:, 1], jj, -factors * diff[:, 1])
    except Exception:
        pass
    return grad

def obj_and_grad(x_flat):
    """Objective and gradient for L-BFGS-B: maximize LP sum of radii."""
    c = x_flat.reshape(N, 2)
    r, s, res = solve_lp_radii(c)
    if s < 1e-3:
        return 1e5, np.zeros_like(x_flat)
    g = compute_lp_gradient(c, res)
    return -s, -g.flatten()

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse initial configurations
    configs = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4],
        [6,6,5,5,4], [5,4,6,6,5], [6,4,5,6,5], [4,5,6,5,6],
        [5,7,5,5,4], [7,5,6,5,3], [5,5,5,5,6], [6,5,5,5,5],
        [4,5,5,6,6], [5,6,6,4,5], [6,6,6,4,4], [5,5,4,6,6]
    ]
    
    for pat in patterns:
        if sum(pat) != N:
            continue
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
            y += np.sqrt(3.0) * r0
        base = np.array(pts[:N])
        
        # Normalize to fit comfortably in [0,1]
        mn = base.min(axis=0)
        mx = base.max(axis=0)
        span = mx - mn + 1e-9
        norm_base = (base - mn) / span * 0.85 + 0.075
        configs.append(norm_base)
        
        for _ in range(3):
            p = norm_base + rng.uniform(-0.02, 0.02, norm_base.shape)
            configs.append(np.clip(p, 0.04, 0.96))
            
    # Add rotated variants to break symmetry
    for angle in np.linspace(0.05, np.pi/4, 4):
        base = configs[0].copy()
        c = base - 0.5
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated = (c @ rot.T) + 0.5
        configs.append(np.clip(rotated, 0.04, 0.96))
        
    # Random dense starts
    for _ in range(8):
        configs.append(rng.uniform(0.12, 0.88, (N, 2)))
        
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * N)
    best_s = -1.0
    best_c = None
    best_r = None
    
    # Phase 2: LP Gradient Ascent on centers
    for cfg in configs:
        c0 = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        try:
            res = minimize(obj_and_grad, c0.flatten(), method='L-BFGS-B', jac=True,
                           bounds=bounds_c, options={'maxiter': 6000, 'ftol': 1e-13})
            if np.isfinite(res.fun):
                c_opt = res.x.reshape(N, 2)
                r_opt, s_opt, _ = solve_lp_radii(c_opt)
                if s_opt > best_s:
                    best_s = s_opt
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            pass
            
    # Phase 3: Coordinate Descent / Jiggle Search
    if best_c is not None:
        curr_c = best_c.copy()
        curr_r = best_r.copy()
        curr_s = best_s
        step = 0.014
        
        for _ in range(4000):
            idx = rng.integers(N)
            old = curr_c[idx].copy()
            move = rng.uniform(-step, step, 2)
            new_pos = np.clip(curr_c[idx] + move, 1e-4, 1.0 - 1e-4)
            curr_c[idx] = new_pos
            
            r_try, s_try, _ = solve_lp_radii(curr_c)
            if s_try > curr_s + 1e-9:
                curr_s = s_try
                curr_r = r_try.copy()
                if curr_s > best_s:
                    best_s = curr_s
                    best_c = curr_c.copy()
                    best_r = curr_r.copy()
                step *= 1.0015
            else:
                curr_c[idx] = old
                if rng.random() < 0.12:
                    step *= 0.96
                    
    # Phase 4: Joint SLSQP Polish
    if best_c is not None:
        for _ in range(10):
            c_pert = np.clip(best_c + rng.uniform(-0.004, 0.004, best_c.shape), 1e-4, 1.0-1e-4)
            r_pert = best_r * 0.98
            v0 = np.concatenate([c_pert[:, 0], c_pert[:, 1], r_pert])
            
            def joint_obj(v):
                return -np.sum(v[2*N:])
            
            def joint_cons(v):
                cx, cy, r = v[:N], v[N:2*N], v[2*N:]
                bc = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
                dx = cx[:, None] - cx[None, :]
                dy = cy[:, None] - cy[None, :]
                d2 = dx**2 + dy**2
                rs = r[:, None] + r[None, :]
                mask = np.triu_indices(N, k=1)
                pc = d2[mask] - rs[mask]**2
                return np.concatenate([bc, pc])
                
            bounds_slqp = [(0.0, 1.0)] * (2*N) + [(1e-7, 0.5)] * N
            try:
                res_j = minimize(joint_obj, v0, method='SLSQP', bounds=bounds_slqp,
                                 constraints={'type': 'ineq', 'fun': joint_cons},
                                 options={'maxiter': 6000, 'ftol': 1e-14})
                if np.isfinite(res_j.fun):
                    c_j = np.column_stack((res_j.x[:N], res_j.x[N:2*N]))
                    r_j, s_j, _ = solve_lp_radii(c_j)
                    if s_j > best_s:
                        best_s = s_j
                        best_c = c_j.copy()
                        best_r = r_j.copy()
            except Exception:
                pass
                
    # Fallback safety net
    if best_c is None:
        best_c = np.column_stack(np.meshgrid(np.linspace(0.15, 0.85, 5), np.linspace(0.15, 0.85, 5))).reshape(-1, 2)
        best_c = np.vstack([best_c, [[0.5, 0.5]]])[:N]
        best_r, best_s, _ = solve_lp_radii(best_c)
        
    # Phase 5: Strict safety scaling to guarantee numerical validity
    scale = 1.0
    for i in range(N):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_r *= scale * 0.9999998
    best_s = float(np.sum(best_r))
    
    return best_c, best_r, best_s
