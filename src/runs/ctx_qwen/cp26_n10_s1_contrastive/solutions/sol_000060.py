# sol_000060 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000035 (state cfcb3616) state=55ceca11 sum of radii=2.621572 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_objective(vars_vec):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_vec[0::3])

def compute_constraints(vars_vec):
    """
    Computes pairwise non-overlap constraints.
    Boundary constraints are handled implicitly by the parameterization.
    Returns array of values that must be >= 0.
    """
    n = N_CIRCLES
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Parameterization ensures r <= x <= 1-r and r <= y <= 1-r
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Squared sum of radii
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    # Extract upper triangular constraints (i < j)
    i_idx, j_idx = np.triu_indices(n, k=1)
    return dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx]

def generate_init_hex(seed, scale=1.0, perturb=0.03):
    np.random.seed(seed)
    pts = []
    r_est = 0.10 * scale
    y = r_est
    row = 0
    while len(pts) < N_CIRCLES:
        x = r_est + (row % 2) * r_est
        while x <= 1.0 - r_est and len(pts) < N_CIRCLES:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    
    pts = np.array(pts[:N_CIRCLES])
    if perturb > 0:
        pts += np.random.normal(0, perturb, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        
    r_est = pts[:, 0].min() if pts.shape[0] > 0 else 0.08
    denom = 1.0 - 2.0 * r_est
    u = np.clip((pts[:, 0] - r_est) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r_est) / denom, 0.0, 1.0)
    
    vars_init = np.empty(N_CIRCLES * 3)
    vars_init[0::3] = r_est
    vars_init[1::3] = u
    vars_init[2::3] = v
    return vars_init

def generate_init_grid(seed, perturb=0.03):
    np.random.seed(seed + 100)
    pts = []
    nx, ny = 5, 6
    margin = 0.05
    dx = (1.0 - 2.0*margin) / max(nx - 1, 1)
    dy = (1.0 - 2.0*margin) / max(ny - 1, 1)
    for i in range(ny):
        for j in range(nx):
            if len(pts) < N_CIRCLES:
                pts.append([margin + j*dx, margin + i*dy])
                
    pts = np.array(pts[:N_CIRCLES])
    if perturb > 0:
        pts += np.random.normal(0, perturb, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        
    r_est = 0.08
    denom = 1.0 - 2.0 * r_est
    u = np.clip((pts[:, 0] - r_est) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r_est) / denom, 0.0, 1.0)
    
    vars_init = np.empty(N_CIRCLES * 3)
    vars_init[0::3] = r_est
    vars_init[1::3] = u
    vars_init[2::3] = v
    return vars_init

def generate_init_random(seed):
    np.random.seed(seed + 200)
    pts = np.random.rand(N_CIRCLES, 2) * 0.8 + 0.1
    r_est = 0.05
    denom = 1.0 - 2.0 * r_est
    u = np.clip((pts[:, 0] - r_est) / denom, 0.0, 1.0)
    v = np.clip((pts[:, 1] - r_est) / denom, 0.0, 1.0)
    
    vars_init = np.empty(N_CIRCLES * 3)
    vars_init[0::3] = r_est
    vars_init[1::3] = u
    vars_init[2::3] = v
    return vars_init

def run_packing():
    n = N_CIRCLES
    bounds = [(1e-6, 0.5), (0.0, 1.0), (0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_vars = None
    best_sum = -np.inf
    
    # Phase 1: Diverse initializations
    inits = []
    for i in range(8):
        inits.append(generate_init_hex(seed=i, perturb=0.03))
    for i in range(5):
        inits.append(generate_init_grid(seed=i, perturb=0.03))
    for i in range(8):
        inits.append(generate_init_random(seed=i))
        
    for x0 in inits:
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                c_val = compute_constraints(res.x)
                if np.min(c_val) >= -1e-7:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Perturbation local search to escape minima
    if best_vars is not None:
        for _ in range(20):
            x_pert = best_vars + np.random.normal(0, 0.004, best_vars.shape)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.0, 1.0)
            x_pert[2::3] = np.clip(x_pert[2::3], 0.0, 1.0)
            x_pert[0::3] = np.clip(x_pert[0::3], 1e-6, 0.5)
            
            try:
                res = minimize(compute_objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    c_val = compute_constraints(res.x)
                    if np.min(c_val) >= -1e-7:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_vars = res.x.copy()
            except Exception:
                continue
                
        # Phase 3: High-precision polish
        try:
            res_final = minimize(compute_objective, best_vars, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if res_final.success:
                c_val = compute_constraints(res_final.x)
                if np.min(c_val) >= -1e-8:
                    best_vars = res_final.x
        except Exception:
            pass
            
    # Fallback
    if best_vars is None:
        r_f = 0.04
        centers_f = np.random.rand(n, 2)
        centers_f = np.clip(centers_f, r_f, 1.0 - r_f)
        best_vars = np.zeros(n * 3)
        best_vars[0::3] = r_f
        best_vars[1::3] = (centers_f[:, 0] - r_f) / (1.0 - 2.0 * r_f)
        best_vars[2::3] = (centers_f[:, 1] - r_f) / (1.0 - 2.0 * r_f)
        
    # Reconstruct centers and radii
    r_opt = best_vars[0::3].copy()
    u_opt = best_vars[1::3].copy()
    v_opt = best_vars[2::3].copy()
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    
    # Strict feasibility cleanup
    for _ in range(3):
        violated = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt((x_opt[i]-x_opt[j])**2 + (y_opt[i]-y_opt[j])**2)
                min_dist = r_opt[i] + r_opt[j] - 1e-10
                if dist < min_dist:
                    shrink = (min_dist - dist) / 2.0 + 1e-12
                    r_opt[i] -= shrink
                    r_opt[j] -= shrink
                    violated = True
        if not violated:
            break
            
    # Update parameters to reflect cleaned radii
    for i in range(n):
        if r_opt[i] < 1e-6: r_opt[i] = 1e-6
        denom = 1.0 - 2.0 * r_opt[i]
        u_opt[i] = np.clip((x_opt[i] - r_opt[i]) / denom, 0.0, 1.0)
        v_opt[i] = np.clip((y_opt[i] - r_opt[i]) / denom, 0.0, 1.0)
        x_opt[i] = r_opt[i] + u_opt[i] * denom
        y_opt[i] = r_opt[i] + v_opt[i] * denom

    centers = np.column_stack((x_opt, y_opt))
    return centers, r_opt, float(np.sum(r_opt))
