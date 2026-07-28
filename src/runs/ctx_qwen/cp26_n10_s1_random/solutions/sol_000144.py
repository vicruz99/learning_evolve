# sol_000144 | problem=circle_packing_26 entrypoint=run_packing
# generation=4 parent=sol_000123 (state 90e3970d) state=34f69cd3 sum of radii=2.607453 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def solve_lp(centers):
    """Solves the LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub_list = []
    b_ub_list = []
    
    # Boundary constraints: r_i <= dist to each wall
    for i in range(n):
        x, y = centers[i]
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_list.append(row); b_ub_list.append(x)
        A_ub_list.append(row); b_ub_list.append(1.0 - x)
        A_ub_list.append(row); b_ub_list.append(y)
        A_ub_list.append(row); b_ub_list.append(1.0 - y)
        
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_list.append(row)
            b_ub_list.append(d)
            
    A_ub = np.array(A_ub_list)
    b_ub = np.array(b_ub_list)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and np.isfinite(res.fun):
            return res.x, -res.fun
    except Exception:
        pass
    return None, 0.0

def obj_equal(x):
    """Objective for equal-radius optimization: maximize t."""
    return -x[-1]

def cons_equal(x):
    """Constraints for equal-radius optimization."""
    c = x[:2 * N].reshape(N, 2)
    t = x[-1]
    
    # Boundary constraints
    cons = np.concatenate([
        c[:, 0] - t,
        1.0 - c[:, 0] - t,
        c[:, 1] - t,
        1.0 - c[:, 1] - t
    ])
    
    # Pairwise distance constraints: ||c_i - c_j||^2 >= 4t^2
    ii, jj = np.triu_indices(N, k=1)
    dist_sq = np.sum((c[ii] - c[jj]) ** 2, axis=1)
    cons = np.concatenate([cons, dist_sq - 4.0 * t * t])
    return cons

def hill_climb_lp(centers, steps=3000):
    """Stochastic hill-climbing on centers, evaluated via LP."""
    n = centers.shape[0]
    current_centers = centers.copy()
    best_centers = current_centers.copy()
    
    _, best_sum = solve_lp(best_centers)
    if best_sum <= 0.0:
        return best_centers, np.zeros(n), 0.0
        
    best_radii = solve_lp(best_centers)[0]
    
    step_size = 0.02
    for step in range(steps):
        # Exponential decay for fine-tuning
        current_step = step_size * (0.9995 ** step)
        
        idx = np.random.randint(n)
        old_pos = current_centers[idx].copy()
        
        # Perturb one random circle
        current_centers[idx] += np.random.uniform(-current_step, current_step, 2)
        current_centers[idx] = np.clip(current_centers[idx], 1e-4, 1.0 - 1e-4)
        
        radii, curr_sum = solve_lp(current_centers)
        if radii is not None and curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = current_centers.copy()
            best_radii = radii.copy()
        else:
            current_centers[idx] = old_pos
            
    return best_centers, best_radii, best_sum

def obj_joint(x):
    """Objective for joint optimization: maximize sum of radii."""
    return -np.sum(x[:N])

def cons_joint(x):
    """Constraints for joint optimization with parameterized boundaries."""
    r = x[:N]
    u = x[N:2 * N]
    v = x[2 * N:3 * N]
    
    # Decode centers
    xc = r + (1.0 - 2.0 * r) * u
    yc = r + (1.0 - 2.0 * r) * v
    c_mat = np.column_stack((xc, yc))
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    ii, jj = np.triu_indices(N, k=1)
    d2 = np.sum((c_mat[ii] - c_mat[jj]) ** 2, axis=1)
    rs = r[ii] + r[jj]
    return d2 - rs ** 2

def run_packing():
    np.random.seed(42)
    best_sum = 0.0
    best_c = None
    best_r = None

    # --- Phase 1: Generate Initial Configurations ---
    starts = []
    r0 = 0.105
    pts = []
    y = r0
    row = 0
    while len(pts) < N:
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 <= 1.0:
            pts.append([x, y])
            x += 2.0 * r0
        y += np.sqrt(3) * r0
        row += 1
    starts.append(np.array(pts[:N]))

    # Perturbed hexagonal lattices
    for _ in range(8):
        p = starts[0].copy()
        p += np.random.uniform(-0.025, 0.025, p.shape)
        p = np.clip(p, 0.05, 0.95)
        starts.append(p)
        
    # Regular grid + center
    gx = np.linspace(0.1, 0.9, 5)
    gy = np.linspace(0.1, 0.9, 5)
    grid = np.array([(x, y) for y in gy for x in gx])
    grid = np.vstack([grid, [0.5, 0.5]])
    starts.append(grid)

    # --- Phase 2: Equal-Radius Optimization ---
    bounds_eq = [(0.0, 1.0)] * (2 * N) + [(0.05, 0.2)]
    best_eq_c = None
    best_eq_t = 0.0

    for cfg in starts:
        x0 = np.concatenate([cfg.flatten(), [0.095]])
        try:
            res = minimize(obj_equal, x0, method='SLSQP',
                           bounds=bounds_eq,
                           constraints={'type': 'ineq', 'fun': cons_equal},
                           options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success and res.x[-1] > best_eq_t:
                best_eq_t = res.x[-1]
                best_eq_c = res.x[:2 * N].reshape(N, 2)
        except Exception:
            pass

    # --- Phase 3: LP Refinement & Hill Climbing ---
    if best_eq_c is not None:
        c_hill, r_hill, s_hill = hill_climb_lp(best_eq_c, steps=4000)
        if s_hill > best_sum:
            best_sum = s_hill
            best_c = c_hill.copy()
            best_r = r_hill.copy()

    # --- Phase 4: Joint Optimization Fine-Tuning ---
    if best_c is not None:
        bounds_joint = [(1e-5, 0.5)] * N + [(0.0, 1.0)] * N + [(0.0, 1.0)] * N
        
        for trial in range(4):
            # Perturb centers slightly
            pert_c = best_c + np.random.uniform(-0.005, 0.005, best_c.shape)
            pert_c = np.clip(pert_c, 0.02, 0.98)
            
            # Map to parameterized variables [r, u, v]
            r_guess = best_r.copy() if best_r is not None else np.full(N, 0.09)
            denom = 1.0 - 2.0 * r_guess
            u = np.clip((pert_c[:, 0] - r_guess) / denom, 0.0, 1.0)
            v = np.clip((pert_c[:, 1] - r_guess) / denom, 0.0, 1.0)
            x0 = np.concatenate([r_guess, u, v])
            
            try:
                res = minimize(obj_joint, x0, method='SLSQP',
                               bounds=bounds_joint,
                               constraints={'type': 'ineq', 'fun': cons_joint},
                               options={'maxiter': 3000, 'ftol': 1e-13})
                if np.isfinite(res.fun):
                    # Decode optimized parameters
                    r_new = res.x[:N]
                    u_new = res.x[N:2 * N]
                    v_new = res.x[2 * N:3 * N]
                    c_new = np.column_stack((r_new + (1.0 - 2.0 * r_new) * u_new,
                                             r_new + (1.0 - 2.0 * r_new) * v_new))
                    
                    # Final LP pass on joint-optimized centers
                    r_lp, s_lp = solve_lp(c_new)
                    if r_lp is not None and s_lp > best_sum:
                        best_sum = s_lp
                        best_c = c_new.copy()
                        best_r = r_lp.copy()
            except Exception:
                pass

    # Fallback if all optimization paths fail
    if best_c is None:
        best_c = starts[0]
        best_r = np.full(N, 0.09)
        best_sum = np.sum(best_r)

    # --- Strict Safety Scaling ---
    # Ensure absolute compliance with 1e-12 numerical tolerance
    scale = 1.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_c[i] - best_c[j])
            rs = best_r[i] + best_r[j]
            if d < rs:
                scale = min(scale, d / rs)
                
    # Also check boundaries explicitly
    for i in range(N):
        x, y, r = best_c[i, 0], best_c[i, 1], best_r[i]
        if r > 1e-9:
            scale = min(scale, x / r, (1.0 - x) / r, y / r, (1.0 - y) / r)
            
    best_r *= scale * 0.999998
    best_sum = np.sum(best_r)
    
    return best_c, best_r, float(best_sum)
