# sol_000087 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000042 (state 26164787) state=1b9594e2 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def get_lp_radii(centers):
    """Given fixed centers, solve LP to exactly maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    x = centers[:, 0]
    y = centers[:, 1]
    bounds_val = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    A_ub.extend(np.eye(n))
    b_ub.extend(bounds_val)
    
    # Pairwise constraints: r_i + r_j <= dist(i, j)
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
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x, -res.fun
        else:
            # Fallback if LP fails (e.g., constraints too tight)
            r_fall = np.min(bounds_val) * 0.9
            return np.full(n, r_fall), n * r_fall
    except Exception:
        return np.full(n, 0.01), n * 0.01

def slsqp_objective(vars_vec):
    """Minimize negative sum of radii."""
    return -np.sum(vars_vec[2*N_CIRCLES:])

def slsqp_constraints(vars_vec):
    """Hard constraints: boundary containment and non-overlap."""
    c = vars_vec[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    r = vars_vec[2*N_CIRCLES:]
    cc = []
    
    # Boundary: x >= r, x <= 1-r, y >= r, y <= 1-r
    cc.append(c[:, 0] - r)
    cc.append(1.0 - c[:, 0] - r)
    cc.append(c[:, 1] - r)
    cc.append(1.0 - c[:, 1] - r)
    
    # Pairwise: dist^2 >= (r_i + r_j)^2
    dx = c[:, 0, np.newaxis] - c[:, 0]
    dy = c[:, 1, np.newaxis] - c[:, 1]
    dsq = dx**2 + dy**2
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    i_idx, j_idx = np.triu_indices(N_CIRCLES, k=1)
    cc.append(dsq[i_idx, j_idx] - rs[i_idx, j_idx]**2)
    
    return np.concatenate(cc)

def run_packing():
    np.random.seed(42)
    bounds_sl = [(0.0, 1.0)] * (2*N_CIRCLES) + [(1e-6, 0.5)] * N_CIRCLES
    cons_sl = {'type': 'ineq', 'fun': slsqp_constraints}
    
    best_sum = -1.0
    best_vars = None
    
    # 1. Generate diverse initial center configurations
    inits_centers = []
    
    # Hexagonal lattice
    pts = []
    r_est = 0.09
    y = r_est
    row = 0
    while len(pts) < N_CIRCLES:
        x_off = (row % 2) * r_est
        x = r_est + x_off
        while x <= 1.0 - r_est and len(pts) < N_CIRCLES:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3) * r_est
        row += 1
    inits_centers.append(np.array(pts[:N_CIRCLES]))
    
    # 5x5 Grid + Center
    g = np.array([[0.1 + i*0.2, 0.1 + j*0.2] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
    inits_centers.append(g)
    
    # Random starts
    for _ in range(15):
        inits_centers.append(np.random.rand(N_CIRCLES, 2) * 0.8 + 0.1)
        
    # 2. Process each initialization: LP -> SLSQP -> LP
    for c0 in inits_centers:
        r0, s0 = get_lp_radii(c0)
        if s0 > best_sum:
            best_sum = s0
            best_vars = np.concatenate([c0.flatten(), r0])
            
        # Shrink radii slightly to guarantee strict feasibility for SLSQP start
        x0 = np.concatenate([c0.flatten(), r0 * 0.95])
        try:
            res = minimize(slsqp_objective, x0, method='SLSQP', bounds=bounds_sl, constraints=cons_sl,
                           options={'maxiter': 2000, 'ftol': 1e-12})
            if res.success:
                c_opt = res.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
                r_opt, s_opt = get_lp_radii(c_opt)
                if s_opt > best_sum:
                    best_sum = s_opt
                    best_vars = np.concatenate([c_opt.flatten(), r_opt])
                    
                # Also consider direct SLSQP result if valid
                if np.min(slsqp_constraints(res.x)) >= -1e-8:
                    s_sl = np.sum(res.x[2*N_CIRCLES:])
                    if s_sl > best_sum:
                        best_sum = s_sl
                        best_vars = res.x
        except Exception:
            continue
            
    # 3. Perturbation loop to escape local minima
    for _ in range(10):
        if best_vars is None: break
        c_cur = best_vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
        
        # Small random perturbation to centers
        pert = np.random.randn(N_CIRCLES, 2) * 0.015
        c_pert = np.clip(c_cur + pert, 0.05, 0.95)
        
        r_p, s_p = get_lp_radii(c_pert)
        if s_p > best_sum:
            best_sum = s_p
            best_vars = np.concatenate([c_pert.flatten(), r_p])
            
        x_p = np.concatenate([c_pert.flatten(), r_p * 0.95])
        try:
            res_p = minimize(slsqp_objective, x_p, method='SLSQP', bounds=bounds_sl, constraints=cons_sl,
                             options={'maxiter': 1500, 'ftol': 1e-12})
            if res_p.success and np.min(slsqp_constraints(res_p.x)) >= -1e-8:
                c_pp = res_p.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
                r_pp, s_pp = get_lp_radii(c_pp)
                if s_pp > best_sum:
                    best_sum = s_pp
                    best_vars = np.concatenate([c_pp.flatten(), r_pp])
        except Exception:
            continue
            
    # 4. Final high-precision polish
    if best_vars is not None:
        x_final = best_vars.copy()
        x_final[2*N_CIRCLES:] *= 0.99
        try:
            res_f = minimize(slsqp_objective, x_final, method='SLSQP', bounds=bounds_sl, constraints=cons_sl,
                             options={'maxiter': 3000, 'ftol': 1e-14})
            if res_f.success:
                c_f = res_f.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
                r_f, s_f = get_lp_radii(c_f)
                if s_f > best_sum:
                    best_sum = s_f
                    best_vars = np.concatenate([c_f.flatten(), r_f])
        except Exception:
            pass
            
    centers = best_vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = best_vars[2*N_CIRCLES:]
    return centers, radii, float(best_sum)
