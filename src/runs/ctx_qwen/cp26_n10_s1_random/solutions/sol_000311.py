# sol_000311 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000100 (state 04884290) state=e69ed3f9 sum of radii=2.581585 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

def solve_lp(centers, n, idx_i, idx_j):
    """Solves LP to maximize sum of radii for fixed centers."""
    # Boundary constraints: r_i <= distance to nearest wall
    wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    bounds = [(0.0, max(w, 1e-12)) for w in wall]
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    dx = centers[idx_i, 0] - centers[idx_j, 0]
    dy = centers[idx_i, 1] - centers[idx_j, 1]
    b_ub = np.hypot(dx, dy)
    
    # Constraint matrix structure is constant for a given n
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def obj_joint(v, n):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def cons_joint(v, n):
    """Inequality constraints >= 0 for valid packing."""
    cx = v[:n]
    cy = v[n:2*n]
    r = v[2*n:]
    
    # Boundary constraints
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    d2 = dx**2 + dy**2
    rs2 = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c = np.concatenate([c, (d2 - rs2)[mask]])
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    idx_i, idx_j = np.triu_indices(n, k=1)
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # 1. Generate diverse initial configurations
    configs = []
    
    # Hexagonal lattice patterns
    row_patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4],
        [5, 6, 4, 6, 5], [6, 5, 5, 6, 4]
    ]
    
    for pat in row_patterns:
        if sum(pat) < n: continue
        pts = []
        y = 0.1
        for ri, cnt in enumerate(pat):
            shift = 0.1 if ri % 2 == 1 else 0.0
            x = 0.1 + shift
            for _ in range(cnt):
                if len(pts) >= n: break
                pts.append([x, y])
                x += 0.2
            y += np.sqrt(3) * 0.1
            if len(pts) >= n: break
        pts = np.array(pts[:n])
        
        # Normalize to fit comfortably inside
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        span = mx - mn
        span[span < 1e-9] = 1.0
        pts_norm = (pts - mn) / span * 0.8 + 0.1
        configs.append(pts_norm)
        
        # Add perturbations
        for _ in range(3):
            configs.append(np.clip(pts_norm + rng.uniform(-0.025, 0.025, (n, 2)), 0.05, 0.95))
            
        # Add rotated versions
        for angle in [0.05, -0.05, 0.15, -0.15]:
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s], [s, c]])
            pts_r = (pts_norm - 0.5) @ rot + 0.5
            configs.append(np.clip(pts_r, 0.05, 0.95))

    # Corner & Edge focused
    cfg_ce = np.zeros((n, 2))
    cfg_ce[0] = [0.1, 0.1]
    cfg_ce[1] = [0.9, 0.1]
    cfg_ce[2] = [0.1, 0.9]
    cfg_ce[3] = [0.9, 0.9]
    for k in range(1, 7):
        cfg_ce[4+k] = [0.1 + k*0.13, 0.1]
    for k in range(1, 7):
        cfg_ce[10+k] = [0.9, 0.1 + k*0.13]
    for k in range(1, 6):
        cfg_ce[16+k] = [0.1 + k*0.15, 0.9]
    cfg_ce[22:] = rng.uniform(0.3, 0.7, (4, 2))
    configs.append(cfg_ce)
    configs.append(np.clip(cfg_ce + rng.uniform(-0.02, 0.02, (n, 2)), 0.05, 0.95))
    
    # Random dense starts
    for _ in range(6):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # 2. Evaluate LP sum for all configs and pick top candidates
    candidates = []
    for cfg in configs:
        r_try, s_try = solve_lp(cfg, n, idx_i, idx_j)
        candidates.append((s_try, cfg))
    candidates.sort(reverse=True, key=lambda x: x[0])
    
    # 3. Joint SLSQP optimization on top candidates
    bounds_vars = [(0.01, 0.99)] * (2*n) + [(1e-6, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': cons_joint, 'args': (n,)}
    
    for i, (s_init, cfg) in enumerate(candidates[:8]): # Top 8
        if s_init <= best_sum: break # Diminishing returns
        
        # Initialize radii reasonably for SLSQP
        lims = np.minimum(np.minimum(cfg[:,0], 1-cfg[:,0]), np.minimum(cfg[:,1], 1-cfg[:,1]))
        diffs = cfg[:, None, :] - cfg[None, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        r0 = np.minimum(lims, np.min(dists, axis=1)/2.0) * 0.85
        r0 = np.maximum(r0, 0.02)
        
        x0 = np.concatenate([cfg[:,0], cfg[:,1], r0])
        
        try:
            res = minimize(obj_joint, x0, args=(n,), method='SLSQP', bounds=bounds_vars,
                           constraints=cons_dict, options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False})
            if np.isfinite(res.fun):
                cx, cy = res.x[:n], res.x[n:2*n]
                c_opt = np.column_stack((cx, cy))
                r_lp, s_lp = solve_lp(c_opt, n, idx_i, idx_j)
                if s_lp > best_sum:
                    best_sum = s_lp
                    best_c = c_opt.copy()
                    best_r = r_lp.copy()
        except Exception:
            continue
            
    # 4. Coordinate Ascent on Centers with LP Oracle
    if best_c is not None:
        c_curr = best_c.copy()
        s_curr = best_sum
        step = 0.035
        no_improve = 0
        max_iter = 2500
        
        for _ in range(max_iter):
            idx = rng.integers(n)
            old_pos = c_curr[idx].copy()
            best_move = old_pos
            best_move_sum = s_curr
            
            # Sample multiple directions
            for _ in range(12):
                trial = old_pos + rng.uniform(-step, step, 2)
                trial = np.clip(trial, 0.02, 0.98)
                c_curr[idx] = trial
                _, s_try = solve_lp(c_curr, n, idx_i, idx_j)
                if s_try > best_move_sum:
                    best_move_sum = s_try
                    best_move = trial.copy()
                    
            c_curr[idx] = best_move
            _, s_curr = solve_lp(c_curr, n, idx_i, idx_j)
            
            if s_curr > best_sum + 1e-8:
                best_sum = s_curr
                best_c = c_curr.copy()
                best_r, _ = solve_lp(best_c, n, idx_i, idx_j)
                no_improve = 0
            else:
                no_improve += 1
                
            step *= 0.9985
            if step < 1e-5: step = 1e-5
            if no_improve > 450: break
            
        # 5. Multi-circle perturbation to break symmetry/local traps
        for _ in range(60):
            c_pert = best_c.copy()
            k = rng.integers(2, 7)
            idxs = rng.choice(n, size=k, replace=False)
            c_pert[idxs] += rng.uniform(-0.025, 0.025, (k, 2))
            c_pert = np.clip(c_pert, 0.05, 0.95)
            r_try, s_try = solve_lp(c_pert, n, idx_i, idx_j)
            
            if s_try > best_sum:
                best_sum = s_try
                best_c = c_pert.copy()
                best_r = r_try.copy()
                
                # Rapid local refinement from perturbed state
                for _ in range(400):
                    i = rng.integers(n)
                    old = best_c[i].copy()
                    best_c[i] += rng.uniform(-0.012, 0.012, 2)
                    best_c[i] = np.clip(best_c[i], 0.02, 0.98)
                    r_t, s_t = solve_lp(best_c, n, idx_i, idx_j)
                    if s_t > best_sum:
                        best_sum = s_t
                        best_r = r_t.copy()
                    else:
                        best_c[i] = old
                        break

    # Fallback safety net
    if best_c is None:
        best_c = configs[0]
        best_r, best_sum = solve_lp(best_c, n, idx_i, idx_j)
        
    # 6. Strict numerical safety scaling
    scale = 1.0
    for i in range(n):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(best_c[i,0]-best_c[j,0], best_c[i,1]-best_c[j,1])
            rs = best_r[i] + best_r[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_r *= scale * 0.9999995
    best_sum = float(np.sum(best_r))
    
    return best_c, best_r, best_sum
