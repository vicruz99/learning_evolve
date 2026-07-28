# sol_000313 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000100 (state 04884290) state=00c90020 sum of radii=1.945454 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_radii_lp(centers):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    # Distance to nearest boundary
    wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    wall = np.maximum(wall, 1e-12)
    
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = centers[idx_i, 0] - centers[idx_j, 0]
    dy = centers[idx_i, 1] - centers[idx_j, 1]
    dists = np.hypot(dx, dy)
    
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    b_ub = dists
    
    c_obj = -np.ones(n)
    bounds = [(0.0, w) for w in wall]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x
    except Exception:
        pass
    return np.full(n, 1e-9)

def joint_constraints(vars, n, idx_i, idx_j):
    """Inequality constraints >= 0 for valid packing."""
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    con = []
    # Boundary
    con.extend(c[:, 0] - r)
    con.extend(1.0 - c[:, 0] - r)
    con.extend(c[:, 1] - r)
    con.extend(1.0 - c[:, 1] - r)
    # Pairwise non-overlap
    dx = c[idx_i, 0] - c[idx_j, 0]
    dy = c[idx_i, 1] - c[idx_j, 1]
    d2 = dx**2 + dy**2
    rs = r[idx_i] + r[idx_j]
    con.extend(d2 - rs**2)
    return np.concatenate(con)

def equal_rad_constraints(vars, n, idx_i, idx_j):
    """Constraints for equal radius optimization."""
    c = vars[:2*n].reshape(n, 2)
    R = vars[2*n]
    con = []
    con.extend(c[:, 0] - R)
    con.extend(1.0 - c[:, 0] - R)
    con.extend(c[:, 1] - R)
    con.extend(1.0 - c[:, 1] - R)
    dx = c[idx_i, 0] - c[idx_j, 0]
    dy = c[idx_i, 1] - c[idx_j, 1]
    d = np.hypot(dx, dy)
    con.extend(d - 2.0 * R)
    return np.concatenate(con)

def generate_configs(n, rng):
    """Generates diverse hexagonal and random configurations."""
    configs = []
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4]
    ]
    for pat in patterns:
        pts = []
        y = 0.09
        for ri, cnt in enumerate(pat):
            shift = 0.09 if ri % 2 == 1 else 0.0
            x = 0.09 + shift
            for _ in range(cnt):
                if len(pts) >= n:
                    break
                pts.append([x, y])
                x += 0.18
            y += 0.156
        pts = np.array(pts[:n])
        mn, mx = pts.min(0), pts.max(0)
        span = mx - mn
        span[span < 1e-9] = 1.0
        pts_norm = (pts - mn) / span * 0.8 + 0.1
        configs.append(pts_norm)
        
        # Rotated variants
        for ang in [0.04, -0.04, 0.08, -0.08]:
            c_center = np.mean(pts_norm, 0)
            pts_c = pts_norm - c_center
            ca, sa = np.cos(ang), np.sin(ang)
            rot = pts_c @ np.array([[ca, -sa], [sa, ca]])
            mn_r, mx_r = rot.min(0), rot.max(0)
            span_r = mx_r - mn_r
            span_r[span_r < 1e-9] = 1.0
            pts_rot = (rot - mn_r) / span_r * 0.8 + 0.1
            configs.append(pts_rot)
            
    # Random dense starts
    for _ in range(12):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    idx_i, idx_j = np.triu_indices(n, k=1)
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    configs = generate_configs(n, rng)
    bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    cons_joint = {'type': 'ineq', 'fun': joint_constraints, 'args': (n, idx_i, idx_j)}
    cons_eq = {'type': 'ineq', 'fun': equal_rad_constraints, 'args': (n, idx_i, idx_j)}
    
    # Phase 1: Equal-radius topology search
    for cfg in configs[:8]:
        x0 = np.concatenate([cfg.flatten(), [0.09]])
        try:
            res = minimize(lambda v: -v[-1], x0, method='SLSQP', bounds=bounds_vars[:2*n+1],
                           constraints=cons_eq, options={'maxiter': 8000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                c_eq = res.x[:2*n].reshape(n, 2)
                r_lp = solve_radii_lp(c_eq)
                s = np.sum(r_lp)
                if s > best_sum:
                    best_sum = s
                    best_centers = c_eq.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 2: Joint SLSQP refinement from LP results and perturbations
    candidates = [best_centers] if best_centers is not None else [configs[0]]
    for _ in range(6):
        if best_centers is not None:
            pert = np.clip(best_centers + rng.uniform(-0.01, 0.01, (n, 2)), 0.05, 0.95)
            candidates.append(pert)
            
    for cfg in candidates:
        r_init = solve_radii_lp(cfg)
        x0 = np.concatenate([cfg.flatten(), r_init])
        try:
            res = minimize(lambda v: -np.sum(v[2*n:]), x0, args=(n,), method='SLSQP',
                           bounds=bounds_vars, constraints=cons_joint,
                           options={'maxiter': 12000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                c_opt = res.x[:2*n].reshape(n, 2)
                r_opt = np.maximum(res.x[2*n:], 1e-9)
                # Re-solve LP for exact radii at these centers
                r_lp = solve_radii_lp(c_opt)
                s = np.sum(r_lp)
                if s > best_sum:
                    best_sum = s
                    best_centers = c_opt.copy()
                    best_radii = r_lp.copy()
        except Exception:
            continue
            
    # Phase 3: LP-based Hill Climbing on Centers
    if best_centers is not None:
        curr_c = best_centers.copy()
        curr_r = best_radii.copy()
        curr_s = best_sum
        step = 0.02
        
        for iteration in range(3000):
            idx = rng.integers(n)
            old_c = curr_c[idx].copy()
            
            # Try multiple moves per iteration
            best_move = old_c
            best_val = curr_s
            
            for _ in range(5):
                trial = old_c + rng.uniform(-step, step, 2)
                trial = np.clip(trial, 1e-4, 1.0 - 1e-4)
                curr_c[idx] = trial
                r_try = solve_radii_lp(curr_c)
                s_try = np.sum(r_try)
                if s_try > best_val:
                    best_val = s_try
                    best_move = trial.copy()
                    
            curr_c[idx] = best_move
            if best_val > curr_s:
                curr_s = best_val
                curr_r = solve_radii_lp(curr_c)
                best_sum = curr_s
                best_centers = curr_c.copy()
                best_radii = curr_r.copy()
                
            step *= 0.998
            step = max(step, 1e-4)
            
    # Fallback
    if best_centers is None:
        best_centers = configs[0]
        best_radii = solve_radii_lp(best_centers)
        best_sum = np.sum(best_radii)
        
    # Phase 4: Strict numerical safety scaling
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], 
                         best_centers[i, 1] - best_centers[j, 1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d / rs)
                
    best_radii *= scale * 0.9999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
