# sol_000140 | problem=circle_packing_26 entrypoint=run_packing
# generation=7 parent=sol_000074 (state ebc36b4a) state=7914466e sum of radii=2.633035 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Given fixed centers, solves LP to find radii that maximize sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    num_con = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((num_con, n))
    b_ub = np.zeros(num_con)
    k = 0
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        for b in [x, 1.0 - x, y, 1.0 - y]:
            A_ub[k, i] = 1.0
            b_ub[k] = b
            k += 1
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
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
            return res.x, -res.fun
    except Exception:
        pass
        
    return np.full(n, 1e-5), 0.0

def objective_func(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[:N])

def constraint_func(params):
    """Inequality constraints: pairwise non-overlap. Boundaries handled by parameterization."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    
    # Parameterization guarantees r <= x <= 1-r and r <= y <= 1-r
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist2 = dx**2 + dy**2
    
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Constraint: dist^2 >= (r_i + r_j)^2
    return dist2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2

def to_params(centers, radii):
    """Convert physical centers/radii to (r, u, v) optimization parameters."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def hex_init(rng, row_counts, rot=0.0, scale=1.0):
    """Generate a hexagonal lattice initialization."""
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
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
    pts += rng.uniform(-0.02, 0.02, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def force_init(rng):
    """Generate a force-directed layout initialization."""
    pts = rng.uniform(0.1, 0.9, (N, 2))
    for _ in range(300):
        f = np.zeros_like(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        d = np.maximum(d, 1e-4)
        f += np.sum((1.0 / d**2)[:, :, np.newaxis] * diff / d[:, :, np.newaxis], axis=1)
        for dim in range(2):
            f[:, dim] += 20.0 * np.maximum(0, 0.1 - pts[:, dim])
            f[:, dim] -= 20.0 * np.maximum(0, pts[:, dim] - 0.9)
        pts += 0.003 * f
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing():
    rng = np.random.default_rng(42)
    bounds = [(1e-6, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -np.inf
    best_p = None
    
    # Generate diverse initial configurations
    inits = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [5,5,5,5,6],
        [6,4,6,5,5], [7,5,5,5,4], [5,7,5,5,4], [4,5,5,5,7],
        [6,6,5,5,4], [5,5,6,5,5], [6,5,5,5,5], [5,5,5,6,5]
    ]
    for p in patterns:
        for _ in range(6):
            inits.append(hex_init(rng, p, rot=rng.uniform(-0.3, 0.3), scale=rng.uniform(0.85, 1.15)))
    for _ in range(12):
        inits.append(force_init(rng))
        
    # Convert inits to optimization parameters
    param_inits = []
    for c in inits:
        r, _ = solve_lp_radii(c)
        r = np.clip(r * 0.98, 1e-6, 0.49)
        param_inits.append(to_params(c, r))
        
    # Phase 1: Broad SLSQP search
    for x0 in param_inits:
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(constraint_func(res.x)) >= -1e-8:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_p = res.x.copy()
        except Exception:
            continue
            
    if best_p is None:
        c_f = hex_init(rng, [6,5,6,5,4])
        r_f, _ = solve_lp_radii(c_f)
        best_p = to_params(c_f, r_f * 0.98)
        best_sum = -objective_func(best_p)
        
    # Phase 2: LP-Driven Coordinate Ascent on Centers
    r_b = best_p[:N]
    u_b = best_p[N:2*N]
    v_b = best_p[2*N:3*N]
    c_best = np.column_stack((r_b + u_b * (1 - 2 * r_b), r_b + v_b * (1 - 2 * r_b)))
    
    curr_c = c_best.copy()
    curr_s = best_sum
    step = 0.025
    
    for _ in range(3000):
        idx = rng.integers(N)
        d = rng.standard_normal(2)
        d /= np.linalg.norm(d)
        nc = curr_c.copy()
        nc[idx] = np.clip(nc[idx] + step * d, 0.02, 0.98)
        _, ns = solve_lp_radii(nc)
        if ns > curr_s + 1e-7:
            curr_c = nc
            curr_s = ns
            if ns > best_sum:
                best_sum = ns
                r_lp, _ = solve_lp_radii(curr_c)
                best_p = to_params(curr_c, r_lp * 0.995)
            step = max(0.001, step * 0.97)
        else:
            step = min(0.05, step * 1.005)
            
    # Phase 3: SLSQP refinement from improved centers
    if best_p is not None:
        x0 = best_p.copy()
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(constraint_func(res.x)) >= -1e-8:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_p = res.x.copy()
        except Exception:
            pass
            
    # Phase 4: Perturbation & SLSQP to escape local minima
    for _ in range(30):
        xp = best_p.copy()
        xp[:N] += rng.uniform(-0.002, 0.002, N)
        xp[N:3*N] += rng.uniform(-0.015, 0.015, 2*N)
        xp[:N] = np.clip(xp[:N], 1e-6, 0.49)
        xp[N:3*N] = np.clip(xp[N:3*N], 0.0, 1.0)
        try:
            res = minimize(objective_func, xp, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(constraint_func(res.x)) >= -1e-8:
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_p = res.x.copy()
        except Exception:
            continue
            
    # Phase 5: High-precision final polish
    try:
        res_f = minimize(objective_func, best_p, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
        if res_f.success and np.min(constraint_func(res_f.x)) >= -1e-8:
            best_p = res_f.x
            best_sum = -res_f.fun
    except Exception:
        pass
        
    # Reconstruct physical centers and radii
    r_opt = best_p[:N]
    u_opt = best_p[N:2*N]
    v_opt = best_p[2*N:3*N]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    radii = np.maximum(r_opt, 0.0)
    
    return centers, radii, float(best_sum)
