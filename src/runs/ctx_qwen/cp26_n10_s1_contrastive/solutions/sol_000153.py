# sol_000153 | problem=circle_packing_26 entrypoint=run_packing
# generation=9 parent=sol_000125 (state 67e5b4e6) state=79a5d48a sum of radii=2.381373 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Given fixed centers, solves LP to find radii maximizing sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    n_con = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((n_con, n))
    b_ub = np.zeros(n_con)
    k = 0
    
    for i in range(n):
        x, y = centers[i]
        bounds_list = [x, 1.0 - x, y, 1.0 - y]
        for b in bounds_list:
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
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def get_params(centers, radii):
    """Convert physical (centers, radii) to optimization parameters (r, u, v)."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, v])

def get_centers_radii(params):
    """Convert optimization parameters (r, u, v) back to physical (centers, radii)."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return np.column_stack((x, y)), r

def constraint_func(params):
    """Inequality constraints: pairwise non-overlap. Boundaries handled by parameterization."""
    r = params[:N]
    u = params[N:2*N]
    v = params[2*N:3*N]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    return dist_sq[I_IDX, J_IDX] - r_sum[I_IDX, J_IDX]**2

def objective(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[:N])

def generate_hex_init(row_counts, scale, rot, jitter, rng_state):
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
        y += r_est * np.sqrt(3.0)
        row += 1
    pts = np.array(pts[:N])
    pts = (pts - 0.5) * scale + 0.5
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        pts = (pts - 0.5) @ np.array([[c, -s], [s, c]]) + 0.5
    pts += rng_state.uniform(-jitter, jitter, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def generate_force_init(rng_state):
    """Generate a force-directed layout initialization."""
    pts = rng_state.uniform(0.15, 0.85, (N, 2))
    for _ in range(300):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=2))
        d = np.maximum(d, 1e-4)
        f = np.sum((1.0 / d**2)[:, :, np.newaxis] * diff / d[:, :, np.newaxis], axis=1)
        for dim in range(2):
            f[:, dim] += 20.0 * np.maximum(0, 0.1 - pts[:, dim])
            f[:, dim] -= 20.0 * np.maximum(0, pts[:, dim] - 0.9)
        pts += 0.003 * f
        pts = np.clip(pts, 0.05, 0.95)
    return pts

def run_packing():
    np.random.seed(42)
    rng = np.random.default_rng(42)
    
    best_params = None
    best_sum = -np.inf
    best_c = None
    best_r = None
    
    # Generate diverse initial configurations
    inits = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [5,5,5,5,6],
        [6,4,6,5,5], [5,6,4,6,5], [4,5,6,5,6], [6,6,5,5,4],
        [5,5,6,5,5], [7,5,5,5,4], [4,5,5,5,7], [5,7,5,5,4],
        [7,6,5,4,4], [6,7,5,5,3], [8,6,5,4,3], [5,5,5,5,5,1]
    ]
    for pat in patterns:
        for _ in range(3):
            sc = rng.uniform(0.85, 1.15)
            rot = rng.uniform(-0.3, 0.3)
            jitt = rng.uniform(0.005, 0.03)
            rng_st = np.random.RandomState(rng.integers(0, 100000))
            inits.append(generate_hex_init(pat, sc, rot, jitt, rng_st))
            
    for _ in range(15):
        rng_st = np.random.RandomState(rng.integers(0, 100000))
        inits.append(generate_force_init(rng_st))
        
    # Phase 1: Evaluate inits with LP to find a strong starting point
    for c in inits:
        r, s = solve_lp_radii(c)
        if s > best_sum:
            best_sum = s
            best_c = c.copy()
            best_r = r.copy()
            
    # Phase 2: LP-driven Simulated Annealing on centers
    if best_c is not None:
        curr_c = best_c.copy()
        curr_r, curr_s = solve_lp_radii(curr_c)
        temp = 0.04
        step = 0.03
        
        for it in range(8000):
            idx = rng.integers(N)
            old_pos = curr_c[idx].copy()
            
            if rng.random() < 0.08:
                curr_c[idx] = rng.uniform(0.05, 0.95, 2)
            else:
                curr_c[idx] += rng.normal(0, step, 2)
                curr_c[idx] = np.clip(curr_c[idx], 0.01, 0.99)
                
            nr, ns = solve_lp_radii(curr_c)
            delta = ns - curr_s
            
            if delta > 0 or (temp > 1e-7 and rng.random() < np.exp(delta / temp)):
                curr_s = ns
                if ns > best_sum:
                    best_sum = ns
                    best_c = curr_c.copy()
                    best_r = nr.copy()
            else:
                curr_c[idx] = old_pos
                
            temp *= 0.9993
            step = max(0.001, step * 0.9995)
            
    # Phase 3: SLSQP refinement with boundary-safe parameterization
    if best_c is not None:
        r_init = np.clip(best_r * 0.99, 1e-6, 0.49)
        x0 = get_params(best_c, r_init)
        bounds_slqp = [(1e-6, 0.5)] * N + [(0.0, 1.0)] * (2 * N)
        cons_slqp = {'type': 'ineq', 'fun': constraint_func}
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_slqp, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(constraint_func(res.x)) >= -1e-8:
                s_val = -res.fun
                if s_val > best_sum:
                    best_sum = s_val
                    best_params = res.x.copy()
        except Exception:
            pass
            
        # Phase 3b: Perturbation & SLSQP to escape local minima
        if best_params is not None:
            for k in range(25):
                xp = best_params.copy()
                xp[:N] += rng.uniform(-0.0015, 0.0015, N)
                xp[N:3*N] += rng.uniform(-0.02, 0.02, 2*N)
                xp[:N] = np.clip(xp[:N], 1e-6, 0.5)
                xp[N:] = np.clip(xp[N:], 0.0, 1.0)
                
                try:
                    res_p = minimize(objective, xp, method='SLSQP', bounds=bounds_slqp,
                                     constraints=cons_slqp, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                    if res_p.success and np.min(constraint_func(res_p.x)) >= -1e-8:
                        s_val_p = -res_p.fun
                        if s_val_p > best_sum:
                            best_sum = s_val_p
                            best_params = res_p.x.copy()
                except Exception:
                    pass
                    
        # Phase 4: High-precision final polish
        if best_params is not None:
            try:
                res_f = minimize(objective, best_params, method='SLSQP', bounds=bounds_slqp,
                                 constraints=cons_slqp, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
                if res_f.success and np.min(constraint_func(res_f.x)) >= -1e-9:
                    best_params = res_f.x
                    best_sum = -res_f.fun
            except Exception:
                pass
                
    # Fallback handling
    if best_params is None and best_c is not None:
        best_params = get_params(best_c, best_r)
    elif best_params is None:
        pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
        pts = np.vstack([pts, [0.5, 0.5]])
        r_f = np.full(N, 0.08)
        best_params = get_params(pts, r_f)
        best_sum = np.sum(r_f)
        
    centers, radii = get_centers_radii(best_params)
    # Final LP to ensure radii are optimal for the exact final centers
    radii_final, sum_final = solve_lp_radii(centers)
    return centers, radii_final, float(sum_final)
