# sol_000243 | problem=circle_packing_26 entrypoint=run_packing
# generation=8 parent=sol_000177 (state 0ce77dda) state=9343cbe9 sum of radii=2.630713 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
triu_i, triu_j = np.triu_indices(N, k=1)

def get_coords(r, u, v):
    """Compute (x, y) from parameterized variables."""
    return r + (1.0 - 2.0 * r) * u, r + (1.0 - 2.0 * r) * v

def param_constraints(vars_arr, triu_i, triu_j):
    """Inequality constraints >= 0 for pairwise non-overlap."""
    r = vars_arr[:N]
    u = vars_arr[N:2*N]
    v = vars_arr[2*N:3*N]
    x, y = get_coords(r, u, v)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    
    return d2[triu_i, triu_j] - rs[triu_i, triu_j]**2

def param_objective(vars_arr):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_arr[:N])

def solve_lp_radii(centers, A_ub_pre, idx_pairs):
    """Solves LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    b_ub = np.empty(len(idx_pairs))
    for k, (i, j) in enumerate(idx_pairs):
        b_ub[k] = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
        
    bounds = []
    for i in range(n):
        x, y = centers[i]
        lim = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(lim, 1e-9)))
        
    try:
        res = linprog(-np.ones(n), A_ub=A_ub_pre, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def make_hex_init(rows, r0=0.09):
    """Generates initial positions on a hexagonal lattice."""
    pts = []
    y = r0
    for ri, cnt in enumerate(rows):
        shift = r0 if ri % 2 == 1 else 0.0
        x = r0 + shift
        for _ in range(cnt):
            if len(pts) >= N:
                break
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3.0) * r0
    while len(pts) < N:
        pts.append([0.5, 0.5])
    return np.array(pts[:N])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    
    # Precompute LP structure for speed
    idx_pairs = list(zip(triu_i, triu_j))
    A_ub_pre = np.zeros((len(idx_pairs), N))
    for k, (i, j) in enumerate(idx_pairs):
        A_ub_pre[k, i] = 1.0
        A_ub_pre[k, j] = 1.0
        
    patterns = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5],
        [6, 6, 6, 4, 4], [4, 5, 6, 5, 6], [5, 4, 6, 5, 6], [5, 6, 6, 5, 4]
    ]
    
    bounds_vars = [(1e-6, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': param_constraints, 'args': (triu_i, triu_j)}
    
    inits = []
    for p in patterns:
        if sum(p) >= N:
            centers = make_hex_init(p)
            r0 = 0.09
            denom = 1.0 - 2.0 * r0
            u = np.clip((centers[:, 0] - r0) / denom, 0.0, 1.0)
            v = np.clip((centers[:, 1] - r0) / denom, 0.0, 1.0)
            x0 = np.concatenate([[r0]*N, u, v])
            inits.append(x0)
            
            # Perturbed version to break symmetry
            centers_p = centers + rng.uniform(-0.015, 0.015, centers.shape)
            centers_p = np.clip(centers_p, 0.05, 0.95)
            u_p = np.clip((centers_p[:, 0] - r0) / denom, 0.0, 1.0)
            v_p = np.clip((centers_p[:, 1] - r0) / denom, 0.0, 1.0)
            inits.append(np.concatenate([[r0]*N, u_p, v_p]))
            
    # Random starts for diversity
    for _ in range(5):
        r_rand = 0.08
        centers_r = rng.uniform(0.15, 0.85, (N, 2))
        denom = 1.0 - 2.0 * r_rand
        u_r = np.clip((centers_r[:, 0] - r_rand) / denom, 0.0, 1.0)
        v_r = np.clip((centers_r[:, 1] - r_rand) / denom, 0.0, 1.0)
        inits.append(np.concatenate([[r_rand]*N, u_r, v_r]))

    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Phase 1: SLSQP optimization from diverse starts
    for x0 in inits:
        try:
            res = minimize(param_objective, x0, method='SLSQP', bounds=bounds_vars,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14})
            if np.isfinite(res.fun):
                r_opt = res.x[:N]
                u_opt = res.x[N:2*N]
                v_opt = res.x[2*N:3*N]
                x_c, y_c = get_coords(r_opt, u_opt, v_opt)
                centers_tmp = np.column_stack((x_c, y_c))
                
                c_vals = param_constraints(res.x, triu_i, triu_j)
                if np.min(c_vals) >= -1e-7:
                    radii_lp, sum_lp = solve_lp_radii(centers_tmp, A_ub_pre, idx_pairs)
                    if sum_lp > best_sum:
                        best_sum = sum_lp
                        best_centers = centers_tmp.copy()
                        best_radii = radii_lp.copy()
        except Exception:
            pass

    if best_centers is None:
        best_centers = make_hex_init([5, 6, 5, 6, 4])
        best_radii, best_sum = solve_lp_radii(best_centers, A_ub_pre, idx_pairs)

    # Phase 2: Hill Climbing on centers with LP oracle
    centers = best_centers.copy()
    radii = best_radii.copy()
    current_sum = best_sum
    
    step = 0.02
    for iteration in range(2000):
        i = rng.integers(N)
        old_pos = centers[i].copy()
        
        best_move = None
        best_move_sum = current_sum
        
        # Try multiple random displacements
        for _ in range(5):
            trial_pos = old_pos + rng.uniform(-step, step, 2)
            trial_pos = np.clip(trial_pos, 1e-4, 1.0 - 1e-4)
            centers[i] = trial_pos
            
            trial_radii, trial_sum = solve_lp_radii(centers, A_ub_pre, idx_pairs)
            if trial_sum > best_move_sum:
                best_move_sum = trial_sum
                best_move = trial_pos.copy()
                
        centers[i] = old_pos
        if best_move is not None:
            centers[i] = best_move
            radii = solve_lp_radii(centers, A_ub_pre, idx_pairs)[0]
            current_sum = best_move_sum
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
                
        step *= 0.9995

    # Phase 3: Final safety scaling to guarantee strict validity
    scale = 1.0
    for i in range(N):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-9:
            scale = min(scale, x/r, (1.0-x)/r, y/r, (1.0-y)/r)
            
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(best_centers[i,0]-best_centers[j,0], best_centers[i,1]-best_centers[j,1])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-9:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.9999999
    final_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, final_sum
