# sol_000197 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000148 (state 41e5ee41) state=20ef424a sum of radii=2.630169 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_constraints(vars_array, triu_i, triu_j):
    """Computes pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2"""
    r = vars_array[:N]
    u = vars_array[N:2*N]
    v = vars_array[2*N:3*N]
    
    # Parameterization automatically satisfies boundary constraints
    x = r + (1.0 - 2.0 * r) * u
    y = r + (1.0 - 2.0 * r) * v
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d2 = dx**2 + dy**2
    
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    
    return d2[triu_i, triu_j] - rs[triu_i, triu_j]**2

def objective(vars_array):
    """Objective: minimize negative sum of radii => Maximize sum of radii"""
    return -np.sum(vars_array[:N])

def solve_radii_lp(centers, A_ub):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    bounds = []
    b_ub = []
    
    # Boundary constraints handled efficiently via bounds
    for i in range(n):
        x, y = centers[i]
        w = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(w, 1e-9)))
        
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            b_ub.append(dist)
            
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
    except Exception:
        pass
    return np.full(n, 1e-6), 0.0

def penalty_objective(vars_flat, n):
    """Penalty-based objective for joint center/radius optimization."""
    c = vars_flat[:2*n].reshape(n, 2)
    r = vars_flat[2*n:]
    
    # Boundary penalty
    wall = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), np.minimum(c[:, 1], 1.0 - c[:, 1]))
    b_pen = np.sum(np.maximum(0.0, r - wall)**2)
    
    # Overlap penalty
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    o_pen = np.sum(np.maximum(0.0, r_sum - dists)**2)
    
    return -np.sum(r) + 10000.0 * (b_pen + o_pen)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(42)
    triu_i, triu_j = np.triu_indices(N, k=1)
    
    # Precompute A_ub for LP (constant structure across center evaluations)
    A_ub = np.zeros((N * (N - 1) // 2, N))
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_ub[k, i] = 1.0
            A_ub[k, j] = 1.0
            k += 1
            
    bounds_vars = [(1e-6, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (triu_i, triu_j)}
    
    inits = []
    
    # Hexagonal lattice patterns with various row distributions
    row_dists = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], [4, 6, 6, 6, 4],
        [5, 7, 5, 5, 4], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4], [5, 6, 6, 5, 4]
    ]
    
    for rd in row_dists:
        pts = []
        y = 0.085
        row_idx = 0
        for count in rd:
            shift = 0.085 if row_idx % 2 == 1 else 0.0
            x = 0.085 + shift
            for _ in range(count):
                if len(pts) < N:
                    pts.append([x, y])
                x += 0.17
            y += 0.147
            row_idx += 1
        if len(pts) == N:
            inits.append(np.array(pts))
            
    # Random dense configurations to escape structural biases
    for _ in range(6):
        inits.append(rng.uniform(0.15, 0.85, (N, 2)))
        
    best_sum = -np.inf
    best_vars = None
    
    # Phase 1: SLSQP optimization from diverse starts
    for cfg in inits:
        r0 = np.full(N, 0.09)
        denom = 1.0 - 2.0 * r0
        u = np.clip((cfg[:, 0] - r0) / denom, 0.0, 1.0)
        v = np.clip((cfg[:, 1] - r0) / denom, 0.0, 1.0)
        x0 = np.concatenate([r0, u, v])
        
        for _ in range(3):
            xp = x0.copy()
            xp[:N] *= rng.uniform(0.95, 1.05, N)
            xp[N:] += rng.uniform(-0.02, 0.02, 2*N)
            xp = np.clip(xp, [1e-6]*N + [0.0]*(2*N), [0.5]*N + [1.0]*(2*N))
            
            try:
                res = minimize(objective, xp, method='SLSQP', bounds=bounds_vars, constraints=cons,
                               options={'maxiter': 4000, 'ftol': 1e-14})
                if np.isfinite(res.fun):
                    c_vals = compute_constraints(res.x, triu_i, triu_j)
                    if np.min(c_vals) > -1e-5:
                        s = -res.fun
                        if s > best_sum:
                            best_sum = s
                            best_vars = res.x.copy()
            except Exception:
                pass
                
    if best_vars is None:
        r0 = np.full(N, 0.09)
        best_vars = np.concatenate([r0, rng.rand(N), rng.rand(N)])
        best_sum = np.sum(r0)

    # Decode SLSQP results to centers
    r_slsqp = best_vars[:N]
    u_out = best_vars[N:2*N]
    v_out = best_vars[2*N:3*N]
    x_out = r_slsqp + (1.0 - 2.0 * r_slsqp) * u_out
    y_out = r_slsqp + (1.0 - 2.0 * r_slsqp) * v_out
    centers_out = np.column_stack((x_out, y_out))
    
    # Phase 2: Exact LP refinement on SLSQP centers
    r_lp, sum_lp = solve_radii_lp(centers_out, A_ub)
    best_radii = r_lp
    best_sum_lp = sum_lp
    
    # Phase 3: Stochastic hill climbing on centers evaluated via LP
    best_centers = centers_out.copy()
    current_radii = r_lp.copy()
    current_sum = sum_lp
    
    for step in range(200):
        i = rng.integers(N)
        old_c = best_centers[i].copy()
        step_size = 0.015 * (0.99 ** step)
        best_centers[i] += rng.uniform(-step_size, step_size, 2)
        best_centers[i] = np.clip(best_centers[i], 1e-4, 1.0 - 1e-4)
        
        r_try, s_try = solve_radii_lp(best_centers, A_ub)
        if s_try > current_sum:
            current_sum = s_try
            current_radii = r_try
        else:
            best_centers[i] = old_c
            
    # Phase 4: Penalty-based L-BFGS-B refinement to escape discrete traps
    bounds_penalty = [(0.05, 0.95)] * (2*N) + [(0.01, 0.5)] * N
    x0_pen = np.concatenate([best_centers.flatten(), current_radii])
    
    try:
        res_pen = minimize(penalty_objective, x0_pen, args=(N,), method='L-BFGS-B', 
                           bounds=bounds_penalty, options={'maxiter': 2000, 'ftol': 1e-15})
        if np.isfinite(res_pen.fun):
            c_pen = res_pen.x[:2*N].reshape(N, 2)
            r_final, s_final = solve_radii_lp(c_pen, A_ub)
            if s_final > current_sum:
                best_centers = c_pen
                current_radii = r_final
                current_sum = s_final
    except Exception:
        pass
        
    # Phase 5: Strict safety scaling to guarantee numerical validity
    scale = 1.0
    for i in range(N):
        x, y = best_centers[i]
        r = current_radii[i]
        if r < 1e-9: continue
        scale = min(scale, x/r, (1.0 - x)/r, y/r, (1.0 - y)/r)
        
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(best_centers[i, 0] - best_centers[j, 0], best_centers[i, 1] - best_centers[j, 1])
            rs = current_radii[i] + current_radii[j]
            if rs < 1e-9: continue
            scale = min(scale, d / rs)
            
    current_radii *= scale * 0.999999
    final_sum = np.sum(current_radii)
    
    return best_centers, current_radii, float(final_sum)
