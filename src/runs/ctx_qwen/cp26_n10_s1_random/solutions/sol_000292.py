# sol_000292 | problem=circle_packing_26 entrypoint=run_packing
# generation=12 parent=sol_000266 (state 46a61c1d) state=65e54a78 sum of radii=2.487718 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def solve_lp_and_gradient(centers):
    """
    Solves LP to maximize sum of radii for fixed centers.
    Returns optimal radii, objective value, and gradient of objective w.r.t centers.
    """
    n = centers.shape[0]
    m = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    k = 0
    for i in range(n):
        x, y = centers[i]
        A_ub[k, i] = 1.0; b_ub[k] = x; k += 1
        A_ub[k, i] = 1.0; b_ub[k] = 1.0 - x; k += 1
        A_ub[k, i] = 1.0; b_ub[k] = y; k += 1
        A_ub[k, i] = 1.0; b_ub[k] = 1.0 - y; k += 1
        
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = centers[idx_i, 0] - centers[idx_j, 0]
    dy = centers[idx_i, 1] - centers[idx_j, 1]
    dists = np.hypot(dx, dy)
    dists = np.maximum(dists, 1e-9)
    
    for p in range(len(idx_i)):
        i, j = idx_i[p], idx_j[p]
        A_ub[4*n + p, i] = 1.0
        A_ub[4*n + p, j] = 1.0
        b_ub[4*n + p] = dists[p]
        
    c_obj = -np.ones(n)
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return None, None, None
        
    if not res.success or not np.isfinite(res.fun):
        return None, None, None
        
    r_opt = res.x
    obj = -res.fun
    
    grad_centers = np.zeros((n, 2))
    try:
        lam = res.ineqlin.marginals
        if lam is not None:
            lam = np.maximum(lam, 0.0)
            pair_lams = lam[4*n:]
            valid = pair_lams > 1e-8
            if np.any(valid):
                v_idx = np.where(valid)[0]
                lams_v = pair_lams[v_idx]
                ii = idx_i[v_idx]
                jj = idx_j[v_idx]
                ds = dists[v_idx]
                factors = (lams_v / ds)[:, np.newaxis]
                diffs = centers[ii] - centers[jj]
                np.add.at(grad_centers, ii, diffs * factors)
                np.add.at(grad_centers, jj, -diffs * factors)
    except AttributeError:
        pass
        
    return r_opt, obj, grad_centers

def force_simulate(centers, steps=300):
    """Vectorized force-directed relaxation to spread circles and avoid overlaps."""
    n = centers.shape[0]
    c = centers.copy()
    vel = np.zeros_like(c)
    dt = 0.003
    damp = 0.85
    r_eff = 0.095
    
    for _ in range(steps):
        forces = np.zeros_like(c)
        dx = c[:, None, 0] - c[None, :, 0]
        dy = c[:, None, 1] - c[None, :, 1]
        dist = np.sqrt(dx**2 + dy**2 + 1e-8)
        np.fill_diagonal(dist, np.inf)
        
        overlap = np.maximum(0.0, 2*r_eff - dist)
        force_mag = overlap * 150.0 / (dist + 1e-8)
        
        f_rep = np.zeros_like(c)
        f_rep[:, 0] += np.sum(dx * force_mag, axis=1)
        f_rep[:, 1] += np.sum(dy * force_mag, axis=1)
        forces += f_rep
        
        forces[:, 0] += np.clip(r_eff - c[:, 0], 0, None) * 200.0
        forces[:, 0] -= np.clip(c[:, 0] - (1.0 - r_eff), 0, None) * 200.0
        forces[:, 1] += np.clip(r_eff - c[:, 1], 0, None) * 200.0
        forces[:, 1] -= np.clip(c[:, 1] - (1.0 - r_eff), 0, None) * 200.0
        
        vel = damp * vel + forces * dt
        c += vel
        c = np.clip(c, 1e-4, 1.0 - 1e-4)
    return c

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.seterr(all='ignore')
    n = 26
    rng = np.random.default_rng(42)
    
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    configs = []
    # Hexagonal lattice inits with varying densities
    for r0 in [0.08, 0.085, 0.09, 0.095, 0.10]:
        pts = []
        y = r0
        row = 0
        while y + r0 < 1.0 and len(pts) < n + 5:
            shift = r0 if row % 2 == 1 else 0.0
            x = r0 + shift
            while x + r0 < 1.0 and len(pts) < n + 5:
                pts.append([x, y])
                x += 2.0 * r0
            y += np.sqrt(3) * r0
            row += 1
        if len(pts) >= n:
            configs.append(np.array(pts[:n]))
            
    # Diverse random starts
    for _ in range(15):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    for cfg in configs:
        # Phase 1: Force simulate to ensure valid, spread-out starting positions
        c_sim = force_simulate(cfg.copy(), steps=300)
        
        c_curr = c_sim.copy()
        r_curr, obj_curr, grad_curr = solve_lp_and_gradient(c_curr)
        if r_curr is None: continue
        
        # Phase 2: LP-Dual Gradient Ascent on centers
        step_size = 0.012
        for it in range(500):
            g_norm = np.linalg.norm(grad_curr)
            if g_norm > 1e-7:
                c_new = c_curr + step_size * grad_curr / g_norm
                c_new = np.clip(c_new, 0.005, 0.995)
                
                r_new, obj_new, grad_new = solve_lp_and_gradient(c_new)
                if r_new is not None and obj_new > obj_curr + 1e-8:
                    c_curr = c_new
                    r_curr = r_new
                    obj_curr = obj_new
                    grad_curr = grad_new
                    step_size = min(step_size * 1.01, 0.04)
                else:
                    step_size *= 0.96
            else:
                # Random perturbation to escape local minima
                if rng.random() < 0.05:
                    c_pert = c_curr + rng.uniform(-0.008, 0.008, (n, 2))
                    c_pert = np.clip(c_pert, 0.05, 0.95)
                    r_pert, obj_pert, g_pert = solve_lp_and_gradient(c_pert)
                    if r_pert is not None and obj_pert > obj_curr:
                        c_curr = c_pert
                        r_curr = r_pert
                        obj_curr = obj_pert
                        grad_curr = g_pert
                        step_size = 0.012
                        
        if obj_curr > best_sum:
            best_sum = obj_curr
            best_centers = c_curr.copy()
            best_radii = r_curr.copy()
            
    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = np.random.uniform(0.2, 0.8, (n, 2))
        best_radii = np.full(n, 0.09)
        best_sum = np.sum(best_radii)
        
    # Phase 3: Strict numerical safety scaling
    scale = 1.0
    c = best_centers
    r = best_radii
    for i in range(n):
        x, y = c[i]
        ri = r[i]
        if ri > 1e-12:
            scale = min(scale, x/ri, (1.0-x)/ri, y/ri, (1.0-y)/ri)
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(c[i,0]-c[j,0], c[i,1]-c[j,1])
            rs = r[i] + r[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    r *= scale * 0.9999995
    final_sum = float(np.sum(r))
    
    return c, r, final_sum
