# sol_000252 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000227 (state bd5d11f3) state=4d09e87d sum of radii=2.617834 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def solve_lp_radii(centers, n):
    """Solves LP to maximize sum of radii for fixed centers."""
    x = centers[:, 0]
    y = centers[:, 1]
    limits = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    limits = np.maximum(limits, 1e-9)
    bounds = [(0.0, lim) for lim in limits]
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = centers[i_idx, 0] - centers[j_idx, 0]
    dy = centers[i_idx, 1] - centers[j_idx, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    m = len(i_idx)
    A_ub = np.zeros((m, n))
    A_ub[np.arange(m), i_idx] = 1.0
    A_ub[np.arange(m), j_idx] = 1.0
    b_ub = dists
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return np.zeros(n), 0.0

def lp_obj_func(vars, n):
    """Objective for center optimization: minimize negative LP sum of radii."""
    centers = vars.reshape(n, 2)
    _, s = solve_lp_radii(centers, n)
    return -s

def get_joint_constraints(vars_arr, n):
    """Computes inequality constraints >= 0 for valid packing."""
    xs = vars_arr[:n]
    ys = vars_arr[n:2*n]
    rs = vars_arr[2*n:]
    c = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dr = rs[i_idx] + rs[j_idx]
    c = np.concatenate([c, dx**2 + dy**2 - dr**2])
    return c

def obj_joint(vars_arr, n):
    """Objective for joint optimization: minimize negative sum of radii."""
    return -np.sum(vars_arr[2*n:])

def make_hex(rows, n, r0):
    """Generates initial positions on a hexagonal lattice."""
    pts = []
    y = r0
    for idx, cnt in enumerate(rows):
        shift = r0 if idx % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) < n:
                pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
    return np.array(pts[:n])

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    
    row_pats = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 5, 5, 6],
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 7, 5, 5, 4],
        [5, 5, 6, 5, 5], [6, 5, 5, 6, 4], [5, 6, 4, 6, 5],
        [6, 6, 6, 4, 4], [5, 5, 5, 5, 5, 1], [7, 5, 6, 5, 3]
    ]
    
    configs = []
    for pat in row_pats:
        if sum(pat) < n: continue
        c = make_hex(pat, n, 0.095)
        configs.append(c)
        configs.append(np.clip(c + rng.uniform(-0.02, 0.02, c.shape), 0.05, 0.95))
        
    for _ in range(6):
        configs.append(rng.uniform(0.15, 0.85, (n, 2)))
        
    # Phase 1: Force-directed expansion simulation
    for cfg in configs:
        c_sim = cfg.copy()
        radii_sim = np.full(n, 0.05)
        vel = np.zeros_like(c_sim)
        k_rep = 300.0
        k_wall = 800.0
        damping = 0.8
        
        for step in range(3000):
            radii_sim *= 1.00008
            
            diff = c_sim[:, None, :] - c_sim[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            dists_safe = np.maximum(dists, 1e-8)
            
            req = radii_sim[:, None] + radii_sim[None, :]
            overlap = np.maximum(0.0, req - dists)
            rep = (overlap * k_rep) / dists_safe
            
            fx = np.sum(diff[:, :, 0] * rep, axis=1)
            fy = np.sum(diff[:, :, 1] * rep, axis=1)
            
            forces = np.column_stack((fx, fy))
            
            w_rep = np.zeros_like(c_sim)
            w_rep[:, 0] += np.maximum(0.0, radii_sim - c_sim[:, 0]) * k_wall
            w_rep[:, 0] -= np.maximum(0.0, c_sim[:, 0] - (1.0 - radii_sim)) * k_wall
            w_rep[:, 1] += np.maximum(0.0, radii_sim - c_sim[:, 1]) * k_wall
            w_rep[:, 1] -= np.maximum(0.0, c_sim[:, 1] - (1.0 - radii_sim)) * k_wall
            
            forces += w_rep
            
            vel = damping * vel + forces * 0.005
            c_sim += vel
            c_sim = np.clip(c_sim, 0.001, 0.999)
            
        r_lp, s_lp = solve_lp_radii(c_sim, n)
        if s_lp > best_sum:
            best_sum = s_lp
            best_centers = c_sim.copy()
            best_radii = r_lp.copy()
            
    # Phase 2: Nelder-Mead optimization on centers using LP objective
    if best_centers is not None:
        x0_nm = best_centers.flatten()
        try:
            res_nm = minimize(lp_obj_func, x0_nm, args=(n,), method='Nelder-Mead',
                              options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-9})
            c_nm = res_nm.x.reshape(n, 2)
            r_nm, s_nm = solve_lp_radii(c_nm, n)
            if s_nm > best_sum:
                best_sum = s_nm
                best_centers = c_nm.copy()
                best_radii = r_nm.copy()
        except Exception:
            pass
            
    # Phase 3: SLSQP joint optimization polish
    if best_centers is not None:
        v0 = np.zeros(3 * n)
        v0[:n] = best_centers[:, 0]
        v0[n:2*n] = best_centers[:, 1]
        v0[2*n:] = best_radii * 0.95
        
        bounds_vars = [(0.0, 1.0)] * (2 * n) + [(1e-6, 0.5)] * n
        cons = {'type': 'ineq', 'fun': get_joint_constraints, 'args': (n,)}
        
        try:
            res_slq = minimize(obj_joint, v0, args=(n,), method='SLSQP', bounds=bounds_vars,
                               constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14})
            if np.isfinite(res_slq.fun):
                c_slq = np.column_stack((res_slq.x[:n], res_slq.x[n:2*n]))
                r_slq, s_slq = solve_lp_radii(c_slq, n)
                if s_slq > best_sum:
                    best_sum = s_slq
                    best_centers = c_slq.copy()
                    best_radii = r_slq.copy()
        except Exception:
            pass
            
    # Fallback if all optimizations fail unexpectedly
    if best_centers is None:
        best_centers = configs[0]
        best_radii, best_sum = solve_lp_radii(best_centers, n)
        
    # Final safety scaling to guarantee strict numerical validity
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = best_centers[i_idx, 0] - best_centers[j_idx, 0]
    dy = best_centers[i_idx, 1] - best_centers[j_idx, 1]
    d = np.sqrt(dx**2 + dy**2)
    rs = best_radii[i_idx] + best_radii[j_idx]
    if np.any(rs > 1e-12):
        scale = min(scale, np.min(d / np.maximum(rs, 1e-12)))
        
    best_radii *= scale * 0.9999999
    best_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, best_sum
