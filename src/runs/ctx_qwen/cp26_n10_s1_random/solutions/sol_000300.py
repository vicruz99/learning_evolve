# sol_000300 | problem=circle_packing_26 entrypoint=run_packing
# generation=13 parent=sol_000289 (state 50cd4c97) state=1865f310 sum of radii=0.126608 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize
import warnings
warnings.filterwarnings('ignore')

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)
M_PAIRS = len(PAIR_I)

def solve_lp(centers):
    """Solves LP to maximize sum of radii given centers. Returns radii, sum, and marginals."""
    bounds = [(0.0, 0.5)] * N
    lims = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    lims = np.maximum(lims, 1e-9)
    
    diffs = centers[PAIR_I] - centers[PAIR_J]
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    
    A_ub = np.zeros((M_PAIRS + N, N))
    b_ub = np.zeros(M_PAIRS + N)
    A_ub[:M_PAIRS, PAIR_I] = 1.0
    A_ub[:M_PAIRS, PAIR_J] = 1.0
    b_ub[:M_PAIRS] = dists
    A_ub[M_PAIRS:, np.arange(N)] = 1.0
    b_ub[M_PAIRS:] = lims
    
    try:
        res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            r = np.maximum(res.x, 0.0)
            s = -res.fun
            try:
                marg = res.marginals.ineqlin
            except AttributeError:
                try:
                    marg = res.ineqlin.marginals
                except AttributeError:
                    marg = None
            return r, s, marg
    except Exception:
        pass
    return np.zeros(N), 0.0, None

def compute_grad(centers, marg):
    """Computes gradient of sum of radii w.r.t centers using LP marginals."""
    grad = np.zeros((N, 2))
    if marg is None:
        return grad
        
    pair_marg = marg[:M_PAIRS]
    wall_marg = marg[M_PAIRS:]
    
    active = pair_marg > 1e-7
    if np.any(active):
        idx = np.where(active)[0]
        lam = pair_marg[active]
        ii, jj = PAIR_I[active], PAIR_J[active]
        d = centers[ii] - centers[jj]
        dist = np.sqrt(np.sum(d**2, axis=1))
        dist = np.where(dist > 1e-9, dist, 1.0)
        vec = (lam[:, np.newaxis] * d / dist[:, np.newaxis])
        np.add.at(grad, ii, vec)
        np.add.at(grad, jj, -vec)
        
    for i in range(N):
        if wall_marg[i] > 1e-7:
            x, y = centers[i]
            lims_i = [x, 1.0 - x, y, 1.0 - y]
            min_l = min(lims_i)
            if abs(min_l - lims_i[0]) < 1e-6:
                grad[i, 0] += wall_marg[i]
            elif abs(min_l - lims_i[1]) < 1e-6:
                grad[i, 0] -= wall_marg[i]
            elif abs(min_l - lims_i[2]) < 1e-6:
                grad[i, 1] += wall_marg[i]
            elif abs(min_l - lims_i[3]) < 1e-6:
                grad[i, 1] -= wall_marg[i]
    return grad

def gradient_ascent(c0):
    """Performs gradient ascent on centers to maximize LP sum of radii."""
    c = c0.copy()
    r, s, marg = solve_lp(c)
    best_c, best_s = c.copy(), s
    step = 0.02
    vel = np.zeros_like(c)
    
    for _ in range(600):
        g = compute_grad(c, marg)
        gn = np.linalg.norm(g)
        if gn < 1e-8:
            break
            
        g_dir = g / gn
        vel = 0.5 * vel + step * g_dir
        c_new = np.clip(c + vel, 1e-4, 1.0 - 1e-4)
        
        r_new, s_new, marg_new = solve_lp(c_new)
        if s_new > best_s + 1e-9:
            c, marg = c_new, marg_new
            best_c, best_s = c.copy(), s_new
            step = min(step * 1.05, 0.06)
        else:
            step *= 0.85
            vel *= 0.6
        if step < 1e-5:
            break
    return best_c, best_s

def joint_slsqp(c0, r0):
    """Joint optimization of centers and radii using SLSQP."""
    n = N
    v0 = np.concatenate([c0.flatten(), r0])
    bounds = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
    
    def obj(v):
        return -np.sum(v[2 * n:])
        
    def cons(v):
        cx, cy, r = v[:n], v[n:2 * n], v[2 * n:]
        c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
        dx = cx[PAIR_I] - cx[PAIR_J]
        dy = cy[PAIR_I] - cy[PAIR_J]
        dr = r[PAIR_I] + r[PAIR_J]
        c = np.concatenate([c, dx**2 + dy**2 - dr**2])
        return c
        
    try:
        res = minimize(obj, v0, method='SLSQP', bounds=bounds,
                       constraints={'type': 'ineq', 'fun': cons},
                       options={'maxiter': 4000, 'ftol': 1e-13})
        if np.isfinite(res.fun):
            return res.x[:n].reshape(n, 2), res.x[2 * n:]
    except Exception:
        pass
    return c0, r0

def generate_configs(rng):
    """Generates diverse initial configurations."""
    configs = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [5,5,6,5,5], [4,6,6,6,4],
        [6,6,5,5,4], [5,4,6,6,5], [5,6,4,6,5], [6,5,5,6,4],
        [7,5,5,5,4], [4,5,7,5,5], [5,5,5,5,6], [6,6,6,4,4],
        [8,6,6,6], [9,5,6,6], [10,8,8], [7,6,6,7]
    ]
    
    for pat in patterns:
        if sum(pat) < N:
            continue
        # Randomized hexagonal
        for _ in range(3):
            r0 = rng.uniform(0.09, 0.105)
            pts = []
            y = r0
            for idx, cnt in enumerate(pat):
                shift = r0 if idx % 2 == 1 else 0.0
                x = r0 + shift
                for _ in range(cnt):
                    if len(pts) >= N:
                        break
                    pts.append([x, y])
                    x += 2.0 * r0
                y += np.sqrt(3) * r0
            cfg = np.array(pts[:N])
            mn, mx = cfg.min(axis=0), cfg.max(axis=0)
            span = mx - mn + 1e-9
            cfg = (cfg - mn) / span * 0.75 + 0.125
            cfg = np.clip(cfg + rng.uniform(-0.01, 0.01, cfg.shape), 0.05, 0.95)
            configs.append(cfg)
            
        # Rotated hexagonal base
        cfg_base_pts = []
        y = 0.1
        for idx, cnt in enumerate(pat):
            shift = 0.1 if idx % 2 == 1 else 0.0
            x = 0.1 + shift
            for _ in range(cnt):
                if len(cfg_base_pts) >= N:
                    break
                cfg_base_pts.append([x, y])
                x += 0.2
            y += np.sqrt(3) * 0.1
        cfg_base = np.array(cfg_base_pts[:N])
        
        for angle in [0.15, -0.15, 0.3, -0.3]:
            cos_t, sin_t = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            cfg_rot = cfg_base @ rot
            mn, mx = cfg_rot.min(axis=0), cfg_rot.max(axis=0)
            span = mx - mn + 1e-9
            cfg_rot = (cfg_rot - mn) / span * 0.75 + 0.125
            cfg_rot = np.clip(cfg_rot + rng.uniform(-0.005, 0.005, cfg_rot.shape), 0.05, 0.95)
            configs.append(cfg_rot)
            
    # Random dense starts
    for _ in range(12):
        configs.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    return configs

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_c, best_r = None, None
    
    configs = generate_configs(rng)
    
    # Phase 1: Gradient ascent from diverse starts
    for cfg in configs:
        c_opt, s_opt = gradient_ascent(cfg)
        if s_opt > best_sum:
            r_lp, best_sum, _ = solve_lp(c_opt)
            best_c, best_r = c_opt.copy(), r_lp.copy()
            
    # Phase 2: Joint SLSQP polish & secondary gradient ascent
    if best_c is not None:
        for _ in range(8):
            c_pert = np.clip(best_c + rng.uniform(-0.006, 0.006, best_c.shape), 0.02, 0.98)
            r_pert = best_r * 0.96
            c_slq, r_slq = joint_slsqp(c_pert, r_pert)
            
            # Verify strict feasibility
            valid = True
            if np.any(c_slq[:, 0] < r_slq - 1e-9) or np.any(c_slq[:, 0] > 1.0 - r_slq + 1e-9): valid = False
            if np.any(c_slq[:, 1] < r_slq - 1e-9) or np.any(c_slq[:, 1] > 1.0 - r_slq + 1e-9): valid = False
            dx = c_slq[PAIR_I, 0] - c_slq[PAIR_J, 0]
            dy = c_slq[PAIR_I, 1] - c_slq[PAIR_J, 1]
            d2 = dx**2 + dy**2
            rs2 = (r_slq[PAIR_I] + r_slq[PAIR_J])**2
            if np.any(d2 < rs2 - 1e-9): valid = False
            
            if valid:
                r_final, s_final, _ = solve_lp(c_slq)
                if r_final is not None and s_final > best_sum:
                    best_sum, best_c, best_r = s_final, c_slq.copy(), r_final.copy()
                    
                c_ga, s_ga = gradient_ascent(c_slq)
                if s_ga > best_sum:
                    r_ga, _, _ = solve_lp(c_ga)
                    best_sum, best_c, best_r = s_ga, c_ga.copy(), r_ga.copy()
                    
    # Phase 3: Final exact LP squeeze
    if best_c is not None:
        r_final, s_final, _ = solve_lp(best_c)
        if r_final is not None and s_final > best_sum:
            best_r, best_sum = r_final, s_final
            
    # Phase 4: Strict Numerical Safety Scaling
    scale = 1.0
    for i in range(N):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    dx = best_c[PAIR_I, 0] - best_c[PAIR_J, 0]
    dy = best_c[PAIR_I, 1] - best_c[PAIR_J, 1]
    d = np.sqrt(dx**2 + dy**2)
    rs = best_r[PAIR_I] + best_r[PAIR_J]
    mask = rs > 1e-12
    if np.any(mask):
        scale = min(scale, np.min(d[mask] / rs[mask]))
        
    best_r *= scale * 0.9999998
    best_sum = float(np.sum(best_r))
    
    return best_c, best_r, best_sum
