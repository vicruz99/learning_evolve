# sol_000302 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000289 (state 50cd4c97) state=4da8a926 sum of radii=0.136000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
M_PAIRS = len(PAIR_I)

def solve_lp_and_grad(centers):
    """Solves LP for max sum of radii given centers, returns radii, sum, and gradient w.r.t centers."""
    n = N
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    
    diffs = centers[PAIR_I] - centers[PAIR_J]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    
    n_constraints = M_PAIRS + 4 * n
    A_ub = np.zeros((n_constraints, n))
    b_ub = np.zeros(n_constraints)
    
    # Pairwise: r_i + r_j <= dists
    A_ub[:M_PAIRS, PAIR_I] = 1.0
    A_ub[:M_PAIRS, PAIR_J] = 1.0
    b_ub[:M_PAIRS] = dists
    
    # Boundaries: r_i <= c_ix, r_i <= 1-c_ix, r_i <= c_iy, r_i <= 1-c_iy
    A_ub[M_PAIRS:M_PAIRS+n, np.arange(n)] = 1.0
    b_ub[M_PAIRS:M_PAIRS+n] = centers[:, 0]
    
    A_ub[M_PAIRS+n:M_PAIRS+2*n, np.arange(n)] = 1.0
    b_ub[M_PAIRS+n:M_PAIRS+2*n] = 1.0 - centers[:, 0]
    
    A_ub[M_PAIRS+2*n:M_PAIRS+3*n, np.arange(n)] = 1.0
    b_ub[M_PAIRS+2*n:M_PAIRS+3*n] = centers[:, 1]
    
    A_ub[M_PAIRS+3*n:M_PAIRS+4*n, np.arange(n)] = 1.0
    b_ub[M_PAIRS+3*n:M_PAIRS+4*n] = 1.0 - centers[:, 1]
    
    bounds = [(0.0, 0.5)] * n
    c_obj = -np.ones(n)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not res.success:
            return None, 0.0, np.zeros((n, 2))
            
        r = np.maximum(res.x, 0.0)
        s = -res.fun
        
        grad = np.zeros((n, 2))
        marg = None
        try:
            marg = res.marginals.ineqlin
        except AttributeError:
            pass
            
        if marg is not None:
            # Pairwise gradients: lambda_ij * (c_i - c_j) / d_ij
            lam = marg[:M_PAIRS]
            mask = lam > 1e-7
            if np.any(mask):
                idx = np.where(mask)[0]
                l = lam[idx]
                ii = PAIR_I[idx]
                jj = PAIR_J[idx]
                d = dists[idx]
                d = np.where(d > 1e-9, d, 1.0)
                vec = l[:, np.newaxis] * (centers[ii] - centers[jj]) / d[:, np.newaxis]
                np.add.at(grad, ii, vec)
                np.add.at(grad, jj, -vec)
                
            # Boundary gradients
            grad[:, 0] += marg[M_PAIRS:M_PAIRS+n]
            grad[:, 0] -= marg[M_PAIRS+n:M_PAIRS+2*n]
            grad[:, 1] += marg[M_PAIRS+2*n:M_PAIRS+3*n]
            grad[:, 1] -= marg[M_PAIRS+3*n:M_PAIRS+4*n]
            
        return r, s, grad
    except Exception:
        return None, 0.0, np.zeros((n, 2))

def gradient_ascent(c0, max_iter=800):
    """Performs gradient ascent on centers to maximize LP sum of radii."""
    c = c0.copy()
    r_best, s_best, _ = solve_lp_and_grad(c)
    if r_best is None:
        return c, s_best, np.zeros(N)
        
    step = 0.012
    vel = np.zeros_like(c)
    
    for _ in range(max_iter):
        r, s, g = solve_lp_and_grad(c)
        if r is None:
            break
            
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-7:
            break
            
        g_dir = g / g_norm
        vel = 0.65 * vel + step * g_dir
        c_new = np.clip(c + vel, 1e-4, 1.0 - 1e-4)
        
        r_new, s_new, _ = solve_lp_and_grad(c_new)
        if r_new is not None and s_new > s_best + 1e-8:
            c = c_new
            r_best, s_best = r_new, s_new
            step = min(step * 1.015, 0.045)
        else:
            step *= 0.92
            vel *= 0.4
            
        if step < 1e-6:
            break
            
    return c, s_best, r_best

def make_hex(row_counts, r0):
    """Generates a hexagonal lattice configuration."""
    pts = []
    y = r0
    for ri, cnt in enumerate(row_counts):
        shift = r0 if ri % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= N:
                break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
    return np.array(pts[:N])

def obj_slqp(v, n):
    """Objective for SLSQP: maximize sum of radii."""
    return -np.sum(v[2*n:])

def cons_slqp(v, n, idx_i, idx_j):
    """Inequality constraints >= 0 for SLSQP joint optimization."""
    cx = v[:n]
    cy = v[n:2*n]
    r = v[2*n:]
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    dx = cx[idx_i] - cx[idx_j]
    dy = cy[idx_i] - cy[idx_j]
    dr = r[idx_i] + r[idx_j]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = 0.0
    best_c = None
    best_r = None
    
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 6, 6, 5, 4],
        [7, 5, 5, 5, 4], [4, 5, 7, 5, 5], [8, 6, 6, 6], 
        [7, 6, 6, 7], [9, 5, 6, 6], [10, 8, 8], [6, 7, 6, 7]
    ]
    
    configs = []
    for pat in patterns:
        if sum(pat) < N:
            continue
        pts = make_hex(pat, 0.10)
        # Center and scale to fit comfortably in [0.15, 0.85]
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = mx - mn + 1e-9
        pts = (pts - mn) / span * 0.68 + 0.16
        configs.append(pts)
        
        # Rotated variants to break symmetry
        for angle in [0.15, -0.15, 0.3, -0.3]:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            pts_rot = pts @ rot
            mn_r, mx_r = pts_rot.min(axis=0), pts_rot.max(axis=0)
            span_r = mx_r - mn_r + 1e-9
            pts_rot = (pts_rot - mn_r) / span_r * 0.68 + 0.16
            configs.append(pts_rot)
            
    for _ in range(8):
        configs.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    # Phase 1: Gradient ascent on centers using LP duals
    for cfg in configs:
        c_opt, s_opt, r_opt = gradient_ascent(cfg)
        if s_opt > best_sum:
            best_sum = s_opt
            best_c = c_opt.copy()
            best_r = r_opt.copy()
            
    # Phase 2: SLSQP polish for precise constraint handling
    if best_c is not None:
        bounds_sl = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
        for _ in range(4):
            c_pert = np.clip(best_c + rng.uniform(-0.002, 0.002, best_c.shape), 0.02, 0.98)
            r_pert = best_r * 0.97
            v0 = np.concatenate([c_pert[:, 0], c_pert[:, 1], r_pert])
            try:
                res = minimize(obj_slqp, v0, args=(N,), method='SLSQP', bounds=bounds_sl,
                               constraints={'type': 'ineq', 'fun': cons_slqp, 'args': (N, PAIR_I, PAIR_J)},
                               options={'maxiter': 5000, 'ftol': 1e-13})
                if np.isfinite(res.fun):
                    c_sl = np.column_stack((res.x[:N], res.x[N:2*N]))
                    r_sl, s_sl, _ = solve_lp_and_grad(c_sl)
                    if r_sl is not None and s_sl > best_sum:
                        best_sum = s_sl
                        best_c = c_sl.copy()
                        best_r = r_sl.copy()
            except Exception:
                continue
                
    # Phase 3: Strict numerical safety scaling
    scale = 1.0
    for i in range(N):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(best_c[i, 0] - best_c[j, 0], best_c[i, 1] - best_c[j, 1])
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_r *= scale * 0.9999995
    best_sum = float(np.sum(best_r))
    
    return best_c, best_r, best_sum
