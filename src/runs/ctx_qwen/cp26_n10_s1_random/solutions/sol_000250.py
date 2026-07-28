# sol_000250 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000238 (state ed5af233) state=61d3a642 sum of radii=2.629090 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import warnings
warnings.filterwarnings('ignore')

def hex_init(n, rows, r0, rng):
    """Generates initial positions on a hexagonal lattice with specified row distribution."""
    pts = []
    y = r0
    for i, cnt in enumerate(rows):
        shift = r0 if i % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= n: break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
    while len(pts) < n:
        pts.append([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)])
    return np.array(pts[:n])

def simulate(centers, radii, steps=1200):
    """Vectorized force-directed simulation to spread circles and grow radii."""
    n = len(radii)
    vel = np.zeros_like(centers)
    dt = 0.006
    damp = 0.85
    for _ in range(steps):
        radii *= 1.00006
        diff = centers[:, None, :] - centers[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e9)
        
        overlap = np.maximum(0.0, radii[:, None] + radii[None, :] - dist)
        force_mag = overlap * 150.0 / (dist + 1e-9)
        f = np.sum(diff * force_mag[:, :, None], axis=1)
        
        # Wall repulsion
        wall_f = np.zeros_like(centers)
        mask_l = centers[:, 0] < radii
        mask_r = centers[:, 0] > 1.0 - radii
        mask_b = centers[:, 1] < radii
        mask_t = centers[:, 1] > 1.0 - radii
        wall_f[mask_l, 0] += (radii[mask_l] - centers[mask_l, 0]) * 300.0
        wall_f[mask_r, 0] -= (centers[mask_r, 0] + radii[mask_r] - 1.0) * 300.0
        wall_f[mask_b, 1] += (radii[mask_b] - centers[mask_b, 1]) * 300.0
        wall_f[mask_t, 1] -= (centers[mask_t, 1] + radii[mask_t] - 1.0) * 300.0
        
        f += wall_f
        vel = damp * vel + f * dt
        centers += vel
        centers = np.clip(centers, 0.001, 0.999)
    return centers, radii

def objective_joint(v, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def constraints_joint(v, n):
    """Inequality constraints >= 0 for valid packing."""
    cx = v[:n]
    cy = v[n:2*n]
    r = v[2*n:]
    c = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    idx = np.triu_indices(n, k=1)
    dx = cx[idx[0]] - cx[idx[1]]
    dy = cy[idx[0]] - cy[idx[1]]
    dr = r[idx[0]] + r[idx[1]]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def lp_radii(centers, A_ub, idx_i, idx_j):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    wall = np.minimum(np.minimum(centers[:,0], 1.0-centers[:,0]), np.minimum(centers[:,1], 1.0-centers[:,1]))
    bounds = [(0.0, max(w, 1e-12)) for w in wall]
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b_ub = dists[idx_i, idx_j]
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success: 
            return res.x, -res.fun
    except: 
        pass
    return np.full(n, 1e-6), 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    rng = np.random.default_rng(42)
    idx_i, idx_j = np.triu_indices(n, k=1)
    m = len(idx_i)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), idx_i] = 1.0
    A_ub[np.arange(m), idx_j] = 1.0
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Diverse structural patterns
    rows_pats = [[6,5,6,5,4], [5,6,5,6,4], [6,6,4,6,4], [4,6,5,6,5], [5,5,6,5,5]]
    configs = []
    for pat in rows_pats:
        for r0 in [0.090, 0.095, 0.100]:
            c = hex_init(n, pat, r0, rng)
            configs.append(c)
            c2 = c + rng.uniform(-0.025, 0.025, c.shape)
            configs.append(np.clip(c2, 0.05, 0.95))
            
    # Phase 1: Simulation + SLSQP from diverse starts
    for cfg in configs:
        c_sim, r_sim = simulate(cfg.copy(), np.full(n, 0.08), steps=800)
        x0 = np.concatenate([c_sim[:,0], c_sim[:,1], np.maximum(r_sim, 1e-4)])
        bounds_opt = [(0.0, 1.0)]*(2*n) + [(1e-5, 0.5)]*n
        
        try:
            res = minimize(objective_joint, x0, args=(n,), method='SLSQP', bounds=bounds_opt,
                           constraints={'type': 'ineq', 'fun': constraints_joint, 'args': (n,)},
                           options={'maxiter': 2000, 'ftol': 1e-12})
            if np.isfinite(res.fun):
                cx = res.x[:n]
                cy = res.x[n:2*n]
                centers_opt = np.column_stack((cx, cy))
                lp_r, lp_s = lp_radii(centers_opt, A_ub, idx_i, idx_j)
                if lp_r is not None and lp_s > best_sum:
                    best_sum = lp_s
                    best_c = centers_opt.copy()
                    best_r = lp_r.copy()
        except: pass

    if best_c is None:
        best_c = configs[0]
        best_r = np.full(n, 0.085)
        best_sum = np.sum(best_r)
        
    # Phase 2: LP-Guided Hill Climbing on Centers
    curr_c = best_c.copy()
    curr_r = best_r.copy()
    curr_s = best_sum
    for step in range(2000):
        scale = 0.015 * (1.0 - step/2000.0)**0.3
        i = rng.integers(n)
        old = curr_c[i].copy()
        curr_c[i] += rng.uniform(-scale, scale, 2)
        curr_c[i] = np.clip(curr_c[i], 1e-4, 1.0-1e-4)
        
        r_new, s_new = lp_radii(curr_c, A_ub, idx_i, idx_j)
        if r_new is not None and s_new > curr_s + 1e-8:
            curr_s = s_new
            curr_r = r_new.copy()
        else:
            curr_c[i] = old
            
    best_c = curr_c
    best_r = curr_r
    best_sum = curr_s
    
    # Phase 3: Strict numerical safety scaling
    scale = 1.0
    for i in range(n):
        x, y, r = best_c[i,0], best_c[i,1], best_r[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
    diff = best_c[:, None, :] - best_c[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, 1e9)
    r_pair = best_r[:, None] + best_r[None, :]
    scale = min(scale, np.min(dists[idx_i, idx_j] / np.maximum(r_pair[idx_i, idx_j], 1e-12)))
    
    best_r *= scale * 0.999999
    best_sum = float(np.sum(best_r))
    
    return best_c, best_r, best_sum
