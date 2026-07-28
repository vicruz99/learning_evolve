# sol_000226 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000188 (state 061cb89c) state=d93a63fe sum of radii=2.413187 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def compute_clearances(centers):
    """Computes distances to boundaries and pairwise distances."""
    n = centers.shape[0]
    b = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                   np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return b, dists

def solve_lp_radii(centers, b, dists):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    bounds = [(0.0, lim) for lim in b]
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    m = len(i_idx)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), i_idx] = 1.0
    A_ub[np.arange(m), j_idx] = 1.0
    b_ub = dists[i_idx, j_idx]
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def penalty_objective(x, n):
    """Smooth penalty objective for joint center/radius optimization."""
    c = x[:2 * n].reshape(n, 2)
    r = x[2 * n:]
    
    bx = np.minimum(c[:, 0], 1.0 - c[:, 0])
    by = np.minimum(c[:, 1], 1.0 - c[:, 1])
    b_pen = np.sum(np.maximum(0.0, r - bx)**2) + np.sum(np.maximum(0.0, r - by)**2)
    
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    o_pen = np.sum(np.maximum(0.0, r_sum - dists)**2)
    
    return -np.sum(r) + 5000.0 * (b_pen + o_pen)

def generate_hex_configs(n):
    """Generates diverse hexagonal lattice initial configurations."""
    configs = []
    row_dists = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4],
        [5, 5, 5, 5, 6], [6, 5, 5, 6, 4], [7, 5, 6, 5, 3],
        [5, 6, 4, 6, 5], [4, 5, 6, 5, 6], [5, 4, 6, 6, 5]
    ]
    
    for rd in row_dists:
        if sum(rd) != n:
            continue
        pts = []
        y = 0.09
        for idx, cnt in enumerate(rd):
            shift = 0.10 if idx % 2 == 1 else 0.0
            width = (cnt - 1) * 0.20
            x_start = 0.5 - width / 2.0 + shift
            for c in range(cnt):
                pts.append([x_start + c * 0.20, y])
            y += 0.173
        configs.append(np.array(pts[:n]))
    return configs

def run_packing():
    n = 26
    rng = np.random.default_rng(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    configs = generate_hex_configs(n)
    for _ in range(6):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for base_cfg in configs:
        cfg = base_cfg + rng.uniform(-0.008, 0.008, (n, 2))
        cfg = np.clip(cfg, 0.05, 0.95)
        
        r0 = np.full(n, 0.08)
        v0 = np.concatenate([cfg.flatten(), r0])
        bounds_opt = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
        
        try:
            res = minimize(penalty_objective, v0, args=(n,), method='L-BFGS-B', 
                           bounds=bounds_opt, options={'maxiter': 15000, 'ftol': 1e-14})
            
            centers = res.x[:2 * n].reshape(n, 2)
            b, d = compute_clearances(centers)
            r_lp, s_lp = solve_lp_radii(centers, b, d)
            
            if r_lp is not None and s_lp > best_sum:
                best_sum = s_lp
                best_centers = centers.copy()
                best_radii = r_lp.copy()
                
                # Hill climbing refinement on centers
                step = 0.012
                for _ in range(500):
                    improved = False
                    for _ in range(25):
                        i = rng.integers(n)
                        old_c = best_centers[i].copy()
                        best_centers[i] += rng.uniform(-step, step, 2)
                        best_centers[i] = np.clip(best_centers[i], 0.02, 0.98)
                        
                        b_new, d_new = compute_clearances(best_centers)
                        r_try, s_try = solve_lp_radii(best_centers, b_new, d_new)
                        
                        if r_try is not None and s_try > best_sum + 1e-7:
                            best_sum = s_try
                            best_radii = r_try
                            improved = True
                            break
                        else:
                            best_centers[i] = old_c
                            
                    if not improved:
                        step *= 0.92
                    if step < 1e-5:
                        break
                        
        except Exception:
            continue
            
    # Fallback configuration
    if best_centers is None:
        best_centers = configs[0]
        b, d = compute_clearances(best_centers)
        r_fb, _ = solve_lp_radii(best_centers, b, d)
        best_radii = r_fb if r_fb is not None else np.full(n, 0.08)
        best_sum = np.sum(best_radii)
        
    # Final strict safety scaling
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    d_pair = np.sqrt((best_centers[i_idx, 0] - best_centers[j_idx, 0])**2 + 
                     (best_centers[i_idx, 1] - best_centers[j_idx, 1])**2)
    rs_pair = best_radii[i_idx] + best_radii[j_idx]
    if np.any(rs_pair > 1e-12):
        scale = min(scale, np.min(d_pair / np.maximum(rs_pair, 1e-12)))
        
    best_radii *= scale * 0.999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
