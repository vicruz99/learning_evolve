# sol_000133 | problem=circle_packing_26 entrypoint=run_packing
# generation=6 parent=sol_000126 (state fbc70012) state=ddcde962 sum of radii=2.627847 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def solve_radii_lp(centers):
    """Given fixed centers, solves LP to find radii that maximize sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    num_constraints = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    
    k = 0
    for i in range(n):
        x, y = centers[i]
        bounds_val = [x, 1.0 - x, y, 1.0 - y]
        for b in bounds_val:
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
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x, -res.fun
    return np.full(n, 1e-5), 1e-4

def objective(vars_vec):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_vec[0::3])

def constraint_func(vars_vec):
    """
    Inequality constraints: dist_sq >= (r_i + r_j)^2.
    Boundary constraints are satisfied by the parameterization.
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
    r_sum_sq = r_sum**2
    
    return dist_sq[I_IDX, J_IDX] - r_sum_sq[I_IDX, J_IDX]

def make_vars_from_lp(centers):
    """Map physical centers to (r, u, v) optimization parameters using LP radii."""
    r, _ = solve_radii_lp(centers)
    r = np.clip(r, 1e-6, 0.499)
    denom = np.clip(1.0 - 2.0 * r, 1e-6, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    v = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    
    vars0 = np.empty(3 * N)
    vars0[0::3] = r
    vars0[1::3] = u
    vars0[2::3] = v
    return vars0

def generate_inits(rng):
    """Generates diverse initial center configurations."""
    inits = []
    
    # 1. Hexagonal lattice patterns with varied row counts, rotations, scales
    patterns = [
        [6,5,6,5,4], [5,6,5,6,4], [4,6,5,6,5],
        [5,5,5,5,6], [6,4,6,5,5], [5,6,4,6,5],
        [4,5,6,5,6], [6,6,5,5,4], [5,5,6,5,5]
    ]
    
    for pat in patterns:
        for _ in range(4):
            pts = []
            r_est = 0.095
            y = r_est
            for r_idx, cnt in enumerate(pat):
                shift = (r_idx % 2) * r_est
                x = r_est + shift
                for _ in range(cnt):
                    if len(pts) < N:
                        pts.append([x, y])
                    x += 2.0 * r_est
                y += r_est * np.sqrt(3.0)
                
            pts = np.array(pts[:N])
            pts -= 0.5
            scale = rng.uniform(0.85, 1.15)
            pts *= scale
            pts += 0.5
            
            rot = rng.uniform(-0.25, 0.25)
            c, s = np.cos(rot), np.sin(rot)
            pts = pts @ np.array([[c, -s], [s, c]])
            
            pts += rng.uniform(-0.015, 0.015, (N, 2))
            pts = np.clip(pts, 0.02, 0.98)
            inits.append(pts)
            
    # 2. Force-directed layouts to find organic packings
    for seed in range(15):
        rng_fd = np.random.RandomState(seed)
        pts = rng_fd.rand(N, 2) * 0.8 + 0.1
        r_curr = np.full(N, 0.05)
        
        for step in range(400):
            diff = pts[:, None, :] - pts[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2)) + 1e-8
            rep = 1.0 / dists**2
            np.fill_diagonal(rep, 0.0)
            forces = np.sum(rep[:, :, None] * diff / dists[:, :, None], axis=1)
            
            for d in range(2):
                forces[:, d] += 25.0 * np.maximum(0, r_curr - pts[:, d])
                forces[:, d] -= 25.0 * np.maximum(0, pts[:, d] - (1.0 - r_curr))
                
            step_size = 0.008 * (0.997**step)
            pts += step_size * forces
            pts = np.clip(pts, 0.02, 0.98)
            
            for i in range(N):
                d_wall = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
                dists_to_others = np.linalg.norm(pts[i] - pts, axis=1)
                dists_to_others[i] = np.inf
                d_pair = np.min(dists_to_others)
                r_curr[i] = 0.9 * min(d_wall, d_pair/2.0)
                
        inits.append(pts)
        
    # 3. Perturbed grids
    for seed in range(10):
        rng_g = np.random.RandomState(seed)
        pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)])
        pts = np.vstack([pts, [0.5, 0.5]])
        pts += rng_g.uniform(-0.03, 0.03, (N, 2))
        pts = np.clip(pts, 0.05, 0.95)
        inits.append(pts)
        
    return inits

def run_packing():
    rng = np.random.default_rng(42)
    bounds = [(1e-6, 0.49), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_vars = None
    best_sum = -np.inf
    best_c = None
    best_r = None
    
    inits = generate_inits(rng)
    
    # Phase 1: Broad search from diverse initializations using SLSQP
    for pts in inits:
        vars0 = make_vars_from_lp(pts)
        # Slight shrink to ensure strict interior feasibility for SLSQP start
        vars0[0::3] *= 0.98
        vars0[0::3] = np.clip(vars0[0::3], 1e-6, 0.49)
        
        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                cons_val = constraint_func(res.x)
                if np.min(cons_val) >= -1e-8:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    if best_vars is not None:
        # Reconstruct best centers for Phase 2
        r_tmp = best_vars[0::3]
        u_tmp = best_vars[1::3]
        v_tmp = best_vars[2::3]
        x_tmp = r_tmp + u_tmp * (1.0 - 2.0 * r_tmp)
        y_tmp = r_tmp + v_tmp * (1.0 - 2.0 * r_tmp)
        best_c = np.column_stack((x_tmp, y_tmp))
        best_r, best_sum = solve_radii_lp(best_c)
        
        # Phase 2: LP-Driven Coordinate Ascent on Centers
        current_c = best_c.copy()
        current_r, curr_sum = best_r, best_sum
        step = 0.025
        
        for it in range(2500):
            idx = rng.integers(N)
            direction = rng.standard_normal(2)
            direction /= np.linalg.norm(direction)
            
            nc = current_c.copy()
            nc[idx] = np.clip(nc[idx] + step * direction, 0.005, 0.995)
            
            nr, ns = solve_radii_lp(nc)
            if ns > curr_sum + 1e-7:
                current_c = nc
                current_r = nr
                curr_sum = ns
                if ns > best_sum:
                    best_c = current_c.copy()
                    best_r = current_r.copy()
                    best_sum = ns
                step = max(0.001, step * 0.96)
            else:
                step = min(0.05, step * 1.01)
                
        # Phase 3: High-precision SLSQP polish on the refined configuration
        vars0 = make_vars_from_lp(best_c)
        vars0[0::3] *= 0.995
        vars0[0::3] = np.clip(vars0[0::3], 1e-6, 0.49)
        
        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                cons_val = constraint_func(res.x)
                if np.min(cons_val) >= -1e-9:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            pass
            
    # Fallback configuration (should rarely be reached)
    if best_vars is None:
        pts = generate_inits(np.random.default_rng(0))[0]
        best_vars = make_vars_from_lp(pts)
        best_sum = np.sum(best_vars[0::3])
        
    # Reconstruct centers from optimized parameters
    r_opt = best_vars[0::3]
    u_opt = best_vars[1::3]
    v_opt = best_vars[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    # Ensure radii are strictly positive and valid
    r_final = np.maximum(r_opt, 0.0)
    
    return centers, r_final, float(best_sum)
