# sol_000102 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000089 (state c83c6c93) state=61650add sum of radii=2.630972 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[0::3])

def constraint_func(vars_vec):
    """
    Computes inequality constraints g(vars_vec) >= 0.
    Uses parameterization to automatically satisfy boundary constraints.
    Only pairwise non-overlap constraints are enforced.
    """
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    return dist_sq[I_IDX, J_IDX] - r_sum[I_IDX, J_IDX]**2

def solve_lp_radii(centers):
    """Given fixed centers, solves the LP to find radii that maximize sum(r_i)."""
    n = centers.shape[0]
    c = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        x, y = centers[i]
        bounds_val = np.array([x, 1.0 - x, y, 1.0 - y])
        for b in bounds_val:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
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
        
    return np.full(n, 1e-5), 2.6e-4

def generate_hex_init(rng, row_counts, rotation=0.0, scale=1.0, jitter_std=0.01):
    """Generate a feasible initial center configuration based on a hexagonal lattice."""
    pts = []
    r_est = 0.095
    y = r_est
    
    for r_idx, cnt in enumerate(row_counts):
        shift = (r_idx % 2) * r_est
        x = r_est + shift
        for _ in range(cnt):
            if len(pts) < N:
                pts.append([x, y])
            x += 2.0 * r_est
        y += r_est * np.sqrt(3.0)
        
    pts = np.array(pts[:N])
    
    # Center and scale
    pts = pts - 0.5
    pts = pts * scale
    pts = pts + 0.5
    
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        rot_mat = np.array([[c, -s], [s, c]])
        pts = pts @ rot_mat.T
        pts = pts - pts.mean(axis=0) + 0.5
        
    pts = pts + rng.normal(0, jitter_std, (N, 2))
    pts = np.clip(pts, 0.02, 0.98)
    return pts

def centers_to_params(centers, radii):
    """Convert centers and radii to the (r, u, v) parameterization."""
    r = radii.copy()
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    
    vars0 = np.empty(3 * N)
    vars0[0::3] = r
    vars0[1::3] = u
    vars0[2::3] = v
    return vars0

def run_packing():
    rng = np.random.default_rng(42)
    bounds = [(1e-6, 0.5), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_vars = None
    best_sum = -np.inf
    
    # Phase 1: Diverse hexagonal lattice initializations
    row_patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [4, 6, 5, 6, 5],
        [5, 5, 5, 5, 6], [6, 4, 6, 5, 5], [5, 6, 4, 6, 5],
        [4, 5, 6, 5, 6], [6, 6, 5, 5, 4], [5, 5, 6, 5, 5],
        [6, 5, 5, 5, 5], [5, 5, 5, 6, 5]
    ]
    
    for pat in row_patterns:
        for _ in range(4):
            rot = rng.uniform(-0.25, 0.25)
            scale = rng.uniform(0.85, 1.15)
            jitter = rng.uniform(0.005, 0.025)
            centers = generate_hex_init(rng, pat, rot, scale, jitter)
            radii, _ = solve_lp_radii(centers)
            radii = np.maximum(radii, 1e-5)
            x0 = centers_to_params(centers, radii * 0.99)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13, 'disp': False})
                if res.success:
                    if np.min(constraint_func(res.x)) >= -1e-7:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_vars = res.x.copy()
            except Exception:
                continue
                
    # Phase 2: Perturbation refinement
    if best_vars is not None:
        for _ in range(40):
            x0 = best_vars.copy()
            # Perturb u, v more aggressively to change topology
            x0[1::3] += rng.normal(0, 0.05, N)
            x0[2::3] += rng.normal(0, 0.05, N)
            x0[0::3] += rng.normal(0, 0.003, N)
            
            x0[0::3] = np.clip(x0[0::3], 1e-6, 0.5)
            x0[1::3] = np.clip(x0[1::3], 0.0, 1.0)
            x0[2::3] = np.clip(x0[2::3], 0.0, 1.0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
                if res.success:
                    if np.min(constraint_func(res.x)) >= -1e-7:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_vars = res.x.copy()
            except Exception:
                continue
                
    # Phase 3: High-precision polish
    if best_vars is not None:
        try:
            res = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if np.min(constraint_func(res.x)) >= -1e-7:
                best_vars = res.x
                best_sum = -res.fun
        except Exception:
            pass
            
    # Fallback
    if best_vars is None:
        centers_f = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
        r_f = np.full(N, 0.09)
        best_vars = centers_to_params(centers_f, r_f)
        best_sum = np.sum(r_f)

    r_opt = best_vars[0::3]
    u_opt = best_vars[1::3]
    v_opt = best_vars[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(best_sum)
