# sol_000163 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000021 (state e4a8cbeb) state=96dfcaa7 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Solves LP to find optimal radii for fixed centers maximizing sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    num_con = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((num_con, n))
    b_ub = np.zeros(num_con)
    k = 0
    
    for i in range(n):
        x, y = centers[i]
        for b in (x, 1.0 - x, y, 1.0 - y):
            A_ub[k, i] = 1.0
            b_ub[k] = b
            k += 1
            
    dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
    dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            b_ub[k] = dists[i, j]
            k += 1
            
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 1e-6

def to_params(centers, radii):
    """Map physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def from_params(params):
    """Reconstruct physical centers and radii from parameters."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return np.column_stack([x, y]), r

def cons_func(params):
    """Inequality constraints: pairwise non-overlap. Boundaries handled by parameterization."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    return d2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def obj_func(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[:N])

def hex_init(rng, row_counts, rot, scale):
    """Generates a hexagonal lattice initialization."""
    pts = []
    r_est = 0.095
    y = r_est
    row = 0
    for cnt in row_counts:
        shift = (row % 2) * r_est
        x = r_est + shift
        for _ in range(cnt):
            if len(pts) < N:
                pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    pts = np.array(pts[:N])
    pts = (pts - 0.5) * scale + 0.5
    if abs(rot) > 1e-6:
        c_val, s_val = np.cos(rot), np.sin(rot)
        M = np.array([[c_val, -s_val], [s_val, c_val]])
        pts = (pts - 0.5) @ M.T + 0.5
    pts += rng.uniform(-0.015, 0.015, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def force_init(seed):
    """Generates a force-directed layout initialization."""
    rng_fd = np.random.RandomState(seed)
    pts = rng_fd.rand(N, 2) * 0.8 + 0.1
    for _ in range(300):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2)) + 1e-4
        f = np.sum((1.0 / d**2)[:, :, np.newaxis] * diff / d[:, :, np.newaxis], axis=1)
        pts += 0.003 * f
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing():
    rng = np.random.default_rng(42)
    bounds_slqp = [(1e-6, 0.49)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons_dict = {'type': 'ineq', 'fun': cons_func}
    
    best_p = None
    best_sum = -np.inf
    best_c = None
    best_r = None
    
    inits = []
    pats = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [7,5,5,5,4], [5,5,5,5,6], 
        [6,6,5,5,4], [5,5,6,5,5], [6,5,5,5,5], [5,5,5,6,5], [6,4,6,5,5],
        [5,7,5,5,4], [4,5,5,5,7], [7,6,5,4,4], [6,7,5,5,3], [8,6,5,4,3]
    ]
    for pat in pats:
        for _ in range(3):
            inits.append(hex_init(rng, pat, rot=rng.uniform(-0.2, 0.2), scale=rng.uniform(0.9, 1.1)))
    for s in range(15):
        inits.append(force_init(s))
        
    init_scores = []
    for c0 in inits:
        r0, s0 = solve_lp_radii(c0)
        init_scores.append((s0, c0, r0))
        
    # Safe sorting without lambda
    scores_only = [item[0] for item in init_scores]
    sorted_indices = np.argsort(scores_only)[::-1]
    init_scores = [init_scores[i] for i in sorted_indices]
    
    # Phase 1: SLSQP on top starts
    for s0, c0, r0 in init_scores[:8]:
        p0 = to_params(c0, np.clip(r0 * 0.995, 1e-6, 0.49))
        try:
            res = minimize(obj_func, p0, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                           options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(cons_func(res.x)) >= -1e-8:
                if -res.fun > best_sum:
                    best_sum = -res.fun
                    best_p = res.x.copy()
        except Exception:
            continue
            
    if best_p is None:
        s0, c0, r0 = init_scores[0]
        best_p = to_params(c0, np.clip(r0 * 0.995, 1e-6, 0.49))
        best_sum = -obj_func(best_p)
        
    best_c, best_r = from_params(best_p)
    
    # Phase 2: LP-driven Simulated Annealing on centers
    curr_c = best_c.copy()
    curr_r, curr_sum = solve_lp_radii(curr_c)
    temp = 0.035
    step = 0.03
    
    for it in range(4000):
        idx = rng.integers(N)
        old = curr_c[idx].copy()
        if rng.random() < 0.05:
            curr_c[idx] = rng.uniform(0.05, 0.95, 2)
        else:
            curr_c[idx] += rng.normal(0, step, 2)
            curr_c[idx] = np.clip(curr_c[idx], 0.01, 0.99)
            
        nr, ns = solve_lp_radii(curr_c)
        delta = ns - curr_sum
        
        if delta > 0 or (temp > 1e-8 and rng.random() < np.exp(delta / temp)):
            curr_sum = ns
            if ns > best_sum:
                best_sum = ns
                best_c = curr_c.copy()
                best_r = nr.copy()
        else:
            curr_c[idx] = old
            
        temp *= 0.9993
        step = max(0.001, step * 0.9994)
        
    if curr_sum > best_sum:
        best_sum = curr_sum
        best_c = curr_c.copy()
        best_r, best_sum = solve_lp_radii(best_c)
        
    # Phase 3: SLSQP perturbations to escape local minima
    best_p = to_params(best_c, np.clip(best_r * 0.995, 1e-6, 0.49))
    for k in range(25):
        xp = best_p.copy()
        xp[:N] += rng.normal(0, 0.002, N)
        xp[N:3*N] += rng.normal(0, 0.02, 2*N)
        xp[:N] = np.clip(xp[:N], 1e-6, 0.49)
        xp[N:3*N] = np.clip(xp[N:3*N], 0.0, 1.0)
        
        try:
            res = minimize(obj_func, xp, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(cons_func(res.x)) >= -1e-8:
                s_val = -res.fun
                if s_val > best_sum:
                    best_sum = s_val
                    best_p = res.x.copy()
        except Exception:
            continue
            
    # Phase 4: Rotation escapes
    for _ in range(10):
        rot = rng.uniform(-0.08, 0.08)
        c_val, s_val = np.cos(rot), np.sin(rot)
        mat = np.array([[c_val, -s_val], [s_val, c_val]])
        c_rot = (best_c - 0.5) @ mat.T + 0.5
        c_rot = np.clip(c_rot, 0.01, 0.99)
        r_rot, _ = solve_lp_radii(c_rot)
        p_rot = to_params(c_rot, np.clip(r_rot * 0.995, 1e-6, 0.49))
        try:
            res_r = minimize(obj_func, p_rot, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                             options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
            if res_r.success and np.min(cons_func(res_r.x)) >= -1e-8:
                s_r = -res_r.fun
                if s_r > best_sum:
                    best_sum = s_r
                    best_p = res_r.x.copy()
        except Exception:
            pass
            
    # Phase 5: High-precision final polish
    try:
        res_f = minimize(obj_func, best_p, method='SLSQP', bounds=bounds_slqp, constraints=cons_dict,
                         options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
        if res_f.success and np.min(cons_func(res_f.x)) >= -1e-9:
            best_p = res_f.x
            best_sum = -res_f.fun
    except Exception:
        pass
        
    centers, radii = from_params(best_p)
    # Final LP to ensure radii are exactly optimal for the final centers
    final_r, final_sum = solve_lp_radii(centers)
    
    return centers, np.maximum(final_r, 0.0), float(final_sum)
