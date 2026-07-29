# sol_000162 | problem=circle_packing_26 entrypoint=run_packing
# generation=11 parent=sol_000021 (state e4a8cbeb) state=21cc6400 sum of radii=2.627816 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import operator

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_lp_radii(centers):
    """Given fixed centers, solves the LP to find radii that maximize sum(r_i)."""
    n = centers.shape[0]
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        for b in (x, 1.0 - x, y, 1.0 - y):
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    dx = centers[:, 0, np.newaxis] - centers[np.newaxis, :, 0]
    dy = centers[:, 1, np.newaxis] - centers[np.newaxis, :, 1]
    dists = np.sqrt(dx**2 + dy**2)
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-5), 1e-4

def objective(params):
    """Objective: minimize negative sum of radii."""
    return -np.sum(params[0::3])

def constraint_func(params):
    """Inequality constraints: dist_sq >= (r_i + r_j)^2. Boundaries handled by parameterization."""
    r = params[0::3]
    u = params[1::3]
    v = params[2::3]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    return dist_sq[I_IDX, J_IDX] - r_sum[I_IDX, J_IDX]**2

def centers_to_params(centers, radii):
    """Convert centers and radii to the (r, u, v) parameterization."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    
    params = np.empty(3 * N)
    params[0::3] = r
    params[1::3] = u
    params[2::3] = v
    return params

def params_to_centers(params):
    """Reconstruct physical centers and radii from parameters."""
    r = params[0::3]
    u = params[1::3]
    v = params[2::3]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    return np.column_stack((x, y)), r

def generate_hex_init(rng, row_counts, rot, scale):
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
        mat = np.array([[c_val, -s_val], [s_val, c_val]])
        pts = (pts - 0.5) @ mat.T + 0.5
        
    pts += rng.uniform(-0.015, 0.015, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def generate_force_init(rng):
    """Generates a force-directed layout initialization."""
    pts = rng.uniform(0.1, 0.9, (N, 2))
    r_curr = np.full(N, 0.05)
    for step in range(300):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-8
        rep = 1.0 / dists**2
        np.fill_diagonal(rep, 0.0)
        forces = np.sum(rep[:, :, np.newaxis] * diff / dists[:, :, np.newaxis], axis=1)
        
        for d in range(2):
            forces[:, d] += 30.0 * np.maximum(0, r_curr - pts[:, d])
            forces[:, d] -= 30.0 * np.maximum(0, pts[:, d] - (1.0 - r_curr))
            
        step_size = 0.008 * (0.992**step)
        pts += step_size * forces
        pts = np.clip(pts, 1e-4, 1.0 - 1e-4)
        
        for i in range(N):
            d_wall = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
            dists_others = np.linalg.norm(pts[i] - pts, axis=1)
            dists_others[i] = np.inf
            d_pair = np.min(dists_others)
            r_curr[i] = 0.75 * min(d_wall, d_pair/2.0)
    return pts

def generate_grid_init(rng):
    """Generates a perturbed grid initialization."""
    pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
    pts = np.vstack([pts, [0.5, 0.5]])
    pts += rng.uniform(-0.03, 0.03, (N, 2))
    return np.clip(pts, 0.05, 0.95)

def run_packing():
    rng = np.random.default_rng(42)
    bounds_slqp = [(1e-6, 0.49), (0.0, 1.0), (0.0, 1.0)] * N
    cons_dict = {'type': 'ineq', 'fun': constraint_func}
    
    best_vars = None
    best_sum = -np.inf
    best_c = None
    best_r = None
    
    inits = []
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5], [5,5,5,5,6],
        [6,4,6,5,5], [5,6,4,6,5], [4,5,6,5,6], [6,6,5,5,4],
        [5,5,6,5,5], [7,5,5,5,4], [4,5,5,5,7], [5,7,5,5,4],
        [7,6,5,4,4], [6,7,5,5,3], [8,6,5,4,3], [5,5,5,5,5,1]
    ]
    
    for pat in patterns:
        for _ in range(3):
            inits.append(generate_hex_init(rng, pat, rot=rng.uniform(-0.3, 0.3), scale=rng.uniform(0.9, 1.1)))
            
    for _ in range(15):
        inits.append(generate_force_init(rng))
        
    for _ in range(10):
        inits.append(generate_grid_init(rng))
        
    # Phase 1: SLSQP from best LP inits
    init_scores = []
    for c0 in inits:
        r0, s0 = solve_lp_radii(c0)
        init_scores.append((s0, c0, r0))
    init_scores.sort(key=operator.itemgetter(0), reverse=True)
    
    for s0, c0, r0 in init_scores[:10]:
        r_safe = np.clip(r0 * 0.99, 1e-6, 0.49)
        p0 = centers_to_params(c0, r_safe)
        try:
            res = minimize(objective, p0, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_dict, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(constraint_func(res.x)) >= -1e-7:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    if best_vars is None:
        c0 = init_scores[0][1]
        r0 = init_scores[0][2] * 0.99
        best_vars = centers_to_params(c0, np.clip(r0, 1e-6, 0.49))
        best_sum = -objective(best_vars)
        
    best_c, best_r = params_to_centers(best_vars)
    
    # Phase 2: LP-driven Simulated Annealing on centers
    curr_c = best_c.copy()
    curr_r, curr_sum = solve_lp_radii(curr_c)
    temp = 0.04
    step = 0.035
    
    for it in range(6000):
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
        
    # Phase 3: SLSQP refinement on SA best
    r_safe = np.clip(best_r * 0.995, 1e-6, 0.49)
    p0 = centers_to_params(best_c, r_safe)
    try:
        res = minimize(objective, p0, method='SLSQP', bounds=bounds_slqp,
                       constraints=cons_dict, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
        if res.success and np.min(constraint_func(res.x)) >= -1e-7:
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_vars = res.x.copy()
                best_c, best_r = params_to_centers(best_vars)
    except Exception:
        pass
        
    # Phase 4: Perturbation & SLSQP to escape local minima
    for k in range(30):
        xp = best_vars.copy()
        xp[0::3] += rng.normal(0, 0.002, N)
        xp[1::3] += rng.normal(0, 0.02, N)
        xp[2::3] += rng.normal(0, 0.02, N)
        xp[0::3] = np.clip(xp[0::3], 1e-6, 0.49)
        xp[1::3] = np.clip(xp[1::3], 0.0, 1.0)
        xp[2::3] = np.clip(xp[2::3], 0.0, 1.0)
        
        try:
            res = minimize(objective, xp, method='SLSQP', bounds=bounds_slqp,
                           constraints=cons_dict, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            if res.success and np.min(constraint_func(res.x)) >= -1e-7:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
                    best_c, best_r = params_to_centers(best_vars)
        except Exception:
            continue
            
    # Phase 5: High-precision final polish
    if best_vars is not None:
        try:
            res_f = minimize(objective, best_vars, method='SLSQP', bounds=bounds_slqp,
                             constraints=cons_dict, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if res_f.success and np.min(constraint_func(res_f.x)) >= -1e-8:
                best_vars = res_f.x
                best_sum = -res_f.fun
                best_c, best_r = params_to_centers(best_vars)
        except Exception:
            pass
            
    # Final LP to ensure radii are optimal for exact centers
    if best_c is not None:
        final_r, final_sum = solve_lp_radii(best_c)
        best_c, best_r, best_sum = best_c, final_r, final_sum
    else:
        best_c = best_vars.reshape(N, 3)[:, :2] if best_vars is not None else np.random.rand(N, 2)
        best_r = best_vars.reshape(N, 3)[:, 2] if best_vars is not None else np.full(N, 0.05)
        best_sum = np.sum(best_r)
        
    return best_c, np.maximum(best_r, 0.0), float(best_sum)
