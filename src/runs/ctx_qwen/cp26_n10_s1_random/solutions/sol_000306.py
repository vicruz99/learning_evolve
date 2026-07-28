# sol_000306 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000286 (state b9c01463) state=ebcf1682 sum of radii=2.210000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog


def solve_lp(centers, n):
    """Solves LP to maximize sum of radii for fixed centers."""
    x, y = centers[:, 0], centers[:, 1]
    lims = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    lims = np.maximum(lims, 1e-9)
    
    c_obj = -np.ones(n)
    bounds = [(0.0, lim) for lim in lims]
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    dx = centers[idx_i, 0] - centers[idx_j, 0]
    dy = centers[idx_i, 1] - centers[idx_j, 1]
    b_ub = np.hypot(dx, dy)
    
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun, res
    except Exception:
        pass
    return None, 0.0, None


def lp_obj_grad(c_flat, n):
    """Objective and gradient for L-BFGS-B optimization of centers."""
    c = c_flat.reshape(n, 2)
    r, s, res_lp = solve_lp(c, n)
    if r is None:
        return -1.0, np.zeros_like(c_flat)
    
    grad_c = np.zeros((n, 2))
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    try:
        marg = np.asarray(res_lp.ineqlin.marginals)
    except AttributeError:
        try:
            marg = np.asarray(res_lp.marginals.ineqlin)
        except AttributeError:
            return -s, np.zeros_like(c_flat)
    
    if marg is not None:
        mask = marg > 1e-9
        idx = np.where(mask)[0]
        lam = marg[idx]
        ii = idx_i[idx]
        jj = idx_j[idx]
        
        dx = c[ii, 0] - c[jj, 0]
        dy = c[ii, 1] - c[jj, 1]
        d = np.hypot(dx, dy)
        d = np.where(d < 1e-12, 1e-12, d)
        
        fx = lam * dx / d
        fy = lam * dy / d
        
        for k in range(len(idx)):
            i, j = ii[k], jj[k]
            grad_c[i, 0] += fx[k]
            grad_c[i, 1] += fy[k]
            grad_c[j, 0] -= fx[k]
            grad_c[j, 1] -= fy[k]
    
    return -s, -grad_c.flatten()


def make_hex(rows, r0, n):
    """Generates a hexagonal lattice with specified row counts."""
    pts = []
    y = r0
    for ri, cnt in enumerate(rows):
        shift = r0 if ri % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n:
                break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
    while len(pts) < n:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])


def rotate_and_center(pts, angle, rng):
    """Rotates points and centers them in [0,1]^2."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    pts = pts @ rot
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn
    span = np.where(span < 1e-9, 1.0, span)
    pts = (pts - mn) / span
    scale = 0.82 + rng.uniform(-0.03, 0.03)
    offset = 0.5 * (1.0 - scale) + rng.uniform(-0.01, 0.01, 2)
    pts = pts * scale + offset
    return np.clip(pts, 0.03, 0.97)


def gradient_ascent(c0, n, rng, max_iter=400):
    """Momentum-based gradient ascent on centers."""
    c = c0.copy()
    r, s, _ = solve_lp(c, n)
    if r is None:
        return c, 0.0
    
    vel = np.zeros_like(c)
    momentum = 0.7
    step = 0.015
    
    for _ in range(max_iter):
        _, _, g = lp_obj_grad(c.flatten(), n)
        g = -g.reshape(n, 2)
        
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-8:
            break
        
        g_dir = g / g_norm
        vel = momentum * vel + step * g_dir
        
        c_new = c + vel
        c_new = np.clip(c_new, 1e-4, 1.0 - 1e-4)
        
        r_new, s_new, _ = solve_lp(c_new, n)
        if r_new is not None and s_new > s + 1e-9:
            c = c_new
            s = s_new
            step = min(step * 1.03, 0.05)
        else:
            step *= 0.8
            vel *= 0.4
        
        if step < 1e-7:
            break
    
    return c, s


def coordinate_ascent(centers, n, rng, max_iter=2000):
    """Coordinate-wise perturbation search."""
    c = centers.copy()
    r, s, _ = solve_lp(c, n)
    if r is None:
        return c, 0.0
    
    step = 0.02
    for it in range(max_iter):
        idx = rng.integers(n)
        old = c[idx].copy()
        
        # Try random perturbation
        move = rng.normal(0, step, 2)
        c[idx] = np.clip(old + move, 1e-4, 1.0 - 1e-4)
        
        r_new, s_new, _ = solve_lp(c, n)
        if r_new is not None and s_new > s + 1e-10:
            s = s_new
            step = min(step * 1.01, 0.03)
        else:
            c[idx] = old
            if rng.random() < 0.05:
                step *= 0.95
        
        # Decay step slowly
        step = max(step * (1.0 - 1e-4), 1e-5)
    
    return c, s


def joint_obj(v, n):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2 * n:])


def joint_cons(v, n):
    """Inequality constraints >= 0 for valid packing."""
    cx = v[:n]
    cy = v[n:2 * n]
    r = v[2 * n:]
    
    cons = np.concatenate([
        cx - r,
        1.0 - cx - r,
        cy - r,
        1.0 - cy - r
    ])
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = cx[idx_i] - cx[idx_j]
    dy = cy[idx_i] - cy[idx_j]
    d2 = dx ** 2 + dy ** 2
    rs = r[idx_i] + r[idx_j]
    cons = np.concatenate([cons, d2 - rs ** 2])
    return cons


def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    
    best_s = -1.0
    best_c = None
    best_r = None
    
    # Generate diverse initial configurations
    configs = []
    
    # Row patterns for hexagonal lattices
    row_patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [6, 6, 5, 5, 4],
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [5, 7, 5, 5, 4],
        [6, 5, 5, 6, 4], [5, 6, 6, 4, 5], [7, 6, 6, 7],
        [8, 6, 6, 6], [5, 5, 5, 5, 6], [6, 5, 5, 5, 5],
        [5, 5, 5, 6, 5], [5, 5, 6, 6, 4], [6, 4, 6, 6, 4],
        [5, 6, 4, 6, 5], [4, 6, 5, 6, 5], [6, 5, 4, 5, 6],
        [5, 6, 5, 5, 5], [4, 5, 7, 5, 5], [5, 5, 7, 5, 4],
        [7, 5, 5, 5, 4], [5, 4, 6, 6, 5], [5, 6, 4, 5, 6]
    ]
    
    for pat in row_patterns:
        if sum(pat) < n:
            continue
        for r0 in [0.085, 0.095, 0.105]:
            pts = make_hex(pat, r0, n)
            # Normalize to fit in [0,1]^2
            mn = pts.min(axis=0)
            mx = pts.max(axis=0)
            span = mx - mn
            span = np.where(span < 1e-9, 1.0, span)
            pts = (pts - mn) / span * 0.82 + 0.09
            configs.append(np.clip(pts, 0.04, 0.96))
            
            # Rotated versions
            for angle in [0.1, -0.1, 0.2, -0.2, 0.05, -0.05]:
                configs.append(rotate_and_center(make_hex(pat, r0, n), angle, rng))
    
    # Corner-biased patterns
    for _ in range(5):
        pts = np.zeros((n, 2))
        # Place some circles in corners
        corners = np.array([[0.12, 0.12], [0.88, 0.12], [0.12, 0.88], [0.88, 0.88]])
        pts[:4] = corners + rng.uniform(-0.02, 0.02, (4, 2))
        pts[4:] = rng.uniform(0.15, 0.85, (n - 4, 2))
        configs.append(np.clip(pts, 0.04, 0.96))
    
    # Random dense starts
    for _ in range(15):
        configs.append(rng.uniform(0.1, 0.9, (n, 2)))
    
    # Grid-based starts
    for spacing in [0.12, 0.14, 0.16]:
        g = np.arange(0.5 - (n ** 0.5 - 0.5) * spacing / 2, 
                       0.5 + (n ** 0.5 - 0.5) * spacing / 2 + spacing, spacing)
        pts = np.array([[x, y] for y in g for x in g])[:n]
        configs.append(np.clip(pts, 0.04, 0.96))
    
    bounds_c = [(1e-4, 1.0 - 1e-4)] * (2 * n)
    
    # Phase 1: Gradient ascent from diverse starts
    for cfg in configs:
        c0 = np.clip(cfg, 1e-4, 1.0 - 1e-4)
        try:
            c_opt, s_opt = gradient_ascent(c0, n, rng)
            if s_opt > best_s:
                r_opt, best_s, _ = solve_lp(c_opt, n)
                if r_opt is not None:
                    best_c = c_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            continue
    
    # Phase 2: Coordinate ascent refinement
    if best_c is not None:
        c_ref, s_ref = coordinate_ascent(best_c, n, rng)
        if s_ref > best_s:
            r_ref, best_s, _ = solve_lp(c_ref, n)
            if r_ref is not None:
                best_c = c_ref.copy()
                best_r = r_ref.copy()
    
    # Phase 3: Multi-scale perturbation with gradient ascent
    if best_c is not None:
        for scale in [0.01, 0.005, 0.002, 0.001]:
            for _ in range(30):
                c_pert = best_c + rng.uniform(-scale, scale, (n, 2))
                c_pert = np.clip(c_pert, 1e-4, 1.0 - 1e-4)
                try:
                    c_p, s_p = gradient_ascent(c_pert, n, rng, max_iter=200)
                    if s_p > best_s + 1e-9:
                        r_p, best_s, _ = solve_lp(c_p, n)
                        if r_p is not None:
                            best_c = c_p.copy()
                            best_r = r_p.copy()
                except Exception:
                    continue
    
    # Phase 4: Another round of coordinate ascent
    if best_c is not None:
        c_ref2, s_ref2 = coordinate_ascent(best_c, n, rng, max_iter=3000)
        if s_ref2 > best_s:
            r_ref2, best_s, _ = solve_lp(c_ref2, n)
            if r_ref2 is not None:
                best_c = c_ref2.copy()
                best_r = r_ref2.copy()
    
    # Phase 5: Joint SLSQP polish
    if best_c is not None:
        for _ in range(5):
            c_pert = best_c + rng.normal(0, 0.001, (n, 2))
            c_pert = np.clip(c_pert, 1e-4, 1.0 - 1e-4)
            v0 = np.concatenate([c_pert[:, 0], c_pert[:, 1], best_r * 0.995])
            
            bounds_slqp = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
            try:
                res = minimize(joint_obj, v0, args=(n,), method='SLSQP',
                               bounds=bounds_slqp,
                               constraints={'type': 'ineq', 'fun': joint_cons, 'args': (n,)},
                               options={'maxiter': 8000, 'ftol': 1e-14})
                if np.isfinite(res.fun):
                    c_j = np.column_stack((res.x[:n], res.x[n:2 * n]))
                    r_j, s_j, _ = solve_lp(c_j, n)
                    if r_j is not None and s_j > best_s:
                        best_s = s_j
                        best_c = c_j.copy()
                        best_r = r_j.copy()
            except Exception:
                continue
    
    # Phase 6: Final fine-grained coordinate ascent
    if best_c is not None:
        c_fin, s_fin = coordinate_ascent(best_c, n, rng, max_iter=5000)
        if s_fin > best_s:
            r_fin, best_s, _ = solve_lp(c_fin, n)
            if r_fin is not None:
                best_c = c_fin.copy()
                best_r = r_fin.copy()
    
    # Fallback if optimization failed
    if best_c is None:
        base = make_hex([6, 5, 6, 5, 4], 0.095, n)
        best_c = np.clip(base, 0.04, 0.96)
        best_r, best_s, _ = solve_lp(best_c, n)
        if best_r is None:
            best_r = np.full(n, 0.08)
            best_s = np.sum(best_r)
    
    # Final safety scaling
    scale = 1.0
    for i in range(n):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_c[i, 0] - best_c[j, 0], best_c[i, 1] - best_c[j, 1])
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
    
    best_r *= scale * 0.9999998
    best_s = float(np.sum(best_r))
    
    return best_c, best_r, best_s
