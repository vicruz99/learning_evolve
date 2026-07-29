# sol_000124 | problem=circle_packing_26 entrypoint=run_packing
# generation=5 parent=sol_000059 (state 3e3cfdc0) state=5039e1c7 sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog, differential_evolution

N = 26

def solve_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to packing constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    m = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((m, n))
    b_ub = np.zeros(m)
    
    k = 0
    for i in range(n):
        x, y = centers[i]
        A_ub[k, i] = 1.0; b_ub[k] = x; k += 1
        A_ub[k, i] = 1.0; b_ub[k] = 1.0 - x; k += 1
        A_ub[k, i] = 1.0; b_ub[k] = y; k += 1
        A_ub[k, i] = 1.0; b_ub[k] = 1.0 - y; k += 1
        
    # Pairwise distance constraints: r_i + r_j <= dist_ij
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = centers[:, 0][:, None] - centers[:, 0][None, :]
    dy = centers[:, 1][:, None] - centers[:, 1][None, :]
    dists = np.sqrt(dx * dx + dy * dy)
    
    for i, j in zip(i_idx, j_idx):
        A_ub[k, i] = 1.0
        A_ub[k, j] = 1.0
        b_ub[k] = dists[i, j]
        k += 1
        
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
    except Exception:
        pass
    return 0.0, np.zeros(n)

def de_objective(centers_flat):
    """Objective for Differential Evolution: negative sum of optimal radii for given centers."""
    centers = centers_flat.reshape(N, 2)
    obj, _ = solve_radii_lp(centers)
    return -obj

def obj_slq(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[:N])

def cons_slq(v):
    """Inequality constraints for SLSQP: pairwise non-overlap. Boundaries handled by parameterization."""
    r = v[:N]
    u = v[N:2 * N]
    w = v[2 * N:3 * N]
    
    # Parameterization guarantees r <= x <= 1-r and r <= y <= 1-r
    x = r + u * (1.0 - 2.0 * r)
    y = r + w * (1.0 - 2.0 * r)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx ** 2 + dy ** 2
    rs = r[:, None] + r[None, :]
    
    i, j = np.triu_indices(N, k=1)
    return d2[i, j] - rs[i, j] ** 2

def get_params(centers, radii):
    """Map physical centers/radii to (r, u, w) optimization parameters."""
    r = radii.copy()
    denom = 1.0 - 2.0 * r
    denom = np.clip(denom, 1e-7, 1.0)
    u = np.clip((centers[:, 0] - r) / denom, 0.0, 1.0)
    w = np.clip((centers[:, 1] - r) / denom, 0.0, 1.0)
    return np.concatenate([r, u, w])

def run_packing():
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Phase 1: Global search on centers using Differential Evolution
    bounds_de = [(0.0, 1.0)] * (2 * N)
    try:
        res_de = differential_evolution(
            func=de_objective,
            bounds=bounds_de,
            popsize=25,
            maxiter=600,
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=42,
            polishing=False,
            workers=1
        )
        best_centers = res_de.x.reshape(N, 2)
        _, best_radii = solve_radii_lp(best_centers)
        best_sum = float(np.sum(best_radii))
    except Exception:
        best_centers = np.random.rand(N, 2)
        _, best_radii = solve_radii_lp(best_centers)
        best_sum = float(np.sum(best_radii))
        
    # Phase 2: SLSQP refinement with boundary-absorbing parameterization
    bounds_slq = [(1e-7, 0.5)] * N + [(0.0, 1.0)] * (2 * N)
    cons_dict = {'type': 'ineq', 'fun': cons_slq}
    
    v0 = get_params(best_centers, best_radii)
    v0[0:N] *= 0.995  # Slight shrink for strict interior feasibility
    try:
        res = minimize(obj_slq, v0, method='SLSQP', bounds=bounds_slq,
                       constraints=cons_dict, options={'maxiter': 5000, 'ftol': 1e-13})
        if res.success and np.min(cons_slq(res.x)) >= -1e-8:
            r_new = res.x[:N]
            u_new = res.x[N:2 * N]
            w_new = res.x[2 * N:3 * N]
            x_new = r_new + u_new * (1.0 - 2.0 * r_new)
            y_new = r_new + w_new * (1.0 - 2.0 * r_new)
            curr_sum = np.sum(r_new)
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = np.column_stack((x_new, y_new))
                best_radii = r_new
    except Exception:
        pass
        
    # Phase 3: Diverse hexagonal & grid restarts
    for seed in range(25):
        rng = np.random.RandomState(seed)
        pts = []
        r_est = 0.095 + rng.uniform(-0.01, 0.01)
        y = r_est
        row = 0
        while len(pts) < N:
            x = r_est + (row % 2) * r_est
            while x <= 1.0 - r_est and len(pts) < N:
                pts.append([x, y])
                x += 2.0 * r_est
            y += np.sqrt(3.0) * r_est
            row += 1
            
        pts = np.array(pts[:N])
        pts += rng.uniform(-0.04, 0.04, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        
        _, r_init = solve_radii_lp(pts)
        r_init = np.maximum(r_init * 0.98, 1e-6)
        v0 = get_params(pts, r_init)
        
        try:
            res = minimize(obj_slq, v0, method='SLSQP', bounds=bounds_slq,
                           constraints=cons_dict, options={'maxiter': 3000, 'ftol': 1e-12})
            if res.success and np.min(cons_slq(res.x)) >= -1e-8:
                curr_sum = np.sum(res.x[:N])
                if curr_sum > best_sum:
                    r_opt = res.x[:N]
                    u_opt = res.x[N:2 * N]
                    w_opt = res.x[2 * N:3 * N]
                    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
                    y_opt = r_opt + w_opt * (1.0 - 2.0 * r_opt)
                    best_sum = curr_sum
                    best_centers = np.column_stack((x_opt, y_opt))
                    best_radii = r_opt
        except Exception:
            pass
            
    # Phase 4: Local perturbation refinement
    rng = np.random.RandomState(123)
    for _ in range(20):
        v_p = get_params(best_centers, best_radii)
        v_p += rng.randn(3 * N) * 0.004
        v_p[0:N] = np.clip(v_p[0:N], 1e-6, 0.5)
        v_p[N:3 * N] = np.clip(v_p[N:3 * N], 0.0, 1.0)
        
        try:
            res = minimize(obj_slq, v_p, method='SLSQP', bounds=bounds_slq,
                           constraints=cons_dict, options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success and np.min(cons_slq(res.x)) >= -1e-8:
                curr_sum = np.sum(res.x[:N])
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    r_opt = res.x[:N]
                    u_opt = res.x[N:2 * N]
                    w_opt = res.x[2 * N:3 * N]
                    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
                    y_opt = r_opt + w_opt * (1.0 - 2.0 * r_opt)
                    best_centers = np.column_stack((x_opt, y_opt))
                    best_radii = r_opt
        except Exception:
            pass

    # Phase 5: High-precision final polish
    if best_centers is not None:
        v_final = get_params(best_centers, best_radii)
        try:
            res_final = minimize(obj_slq, v_final, method='SLSQP', bounds=bounds_slq,
                                 constraints=cons_dict, options={'maxiter': 8000, 'ftol': 1e-14})
            if res_final.success and np.min(cons_slq(res_final.x)) >= -1e-9:
                r_f = res_final.x[:N]
                u_f = res_final.x[N:2 * N]
                w_f = res_final.x[2 * N:3 * N]
                x_f = r_f + u_f * (1.0 - 2.0 * r_f)
                y_f = r_f + w_f * (1.0 - 2.0 * r_f)
                best_sum = float(np.sum(r_f))
                best_centers = np.column_stack((x_f, y_f))
                best_radii = r_f
        except Exception:
            pass

    return best_centers, best_radii, float(best_sum)
