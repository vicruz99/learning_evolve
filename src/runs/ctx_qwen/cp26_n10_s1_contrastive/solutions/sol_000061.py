# sol_000061 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000035 (state cfcb3616) state=2f8de56c sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[0::3])

def constraint_func(vars_vec):
    """
    Computes pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2.
    Boundary constraints are handled automatically by the (r, u, v) parameterization.
    """
    n = 26
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Map normalized u, v in [0,1] to actual coordinates inside [0,1]^2
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    return dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx]

def get_hex_init(n, r_est):
    """Generates a hexagonal lattice initialization."""
    pts = []
    y = r_est
    row = 0
    while len(pts) < n:
        x_start = r_est if row % 2 == 0 else 2.0 * r_est
        x = x_start
        while x <= 1.0 - r_est and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    return np.array(pts[:n])

def get_grid_init(n, r_est):
    """Generates a square grid initialization."""
    pts = []
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    step_x = (1.0 - 2.0 * r_est) / max(cols - 1, 1)
    step_y = (1.0 - 2.0 * r_est) / max(rows - 1, 1)
    for i in range(rows):
        for j in range(cols):
            if len(pts) < n:
                pts.append([r_est + j * step_x, r_est + i * step_y])
    return np.array(pts[:n])

def run_packing():
    n = 26
    best_sol = None
    best_val = -np.inf
    bounds = [(1e-5, 0.5), (0.0, 1.0), (0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    inits = []
    
    # Phase 1: Generate diverse initial configurations
    for seed in range(50):
        np.random.seed(seed)
        init_type = seed % 4
        
        r_est = 0.08 + 0.04 * np.random.rand()
        
        if init_type == 0:
            positions = get_hex_init(n, r_est)
        elif init_type == 1:
            positions = get_grid_init(n, r_est)
        else:
            positions = np.random.uniform(r_est, 1.0 - r_est, (n, 2))
            
        # Add controlled perturbation to break symmetry
        positions += np.random.uniform(-0.02, 0.02, (n, 2))
        positions = np.clip(positions, r_est, 1.0 - r_est)
        
        # Convert physical positions to normalized u, v coordinates
        denom = 1.0 - 2.0 * r_est
        u_init = (positions[:, 0] - r_est) / denom
        v_init = (positions[:, 1] - r_est) / denom
        u_init = np.clip(u_init, 0.0, 1.0)
        v_init = np.clip(v_init, 0.0, 1.0)
        
        vars0 = np.empty(n * 3)
        vars0[0::3] = r_est
        vars0[1::3] = u_init
        vars0[2::3] = v_init
        inits.append(vars0)
        
    # Phase 2: Run SLSQP from each initialization
    for idx, vars0 in enumerate(inits):
        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-13, 'iprint': -1})
            
            cons_val = constraint_func(res.x)
            if np.min(cons_val) >= -1e-7:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_sol = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization fails completely
    if best_sol is None:
        r_f = 0.04
        centers_f = np.random.uniform(r_f, 1.0 - r_f, (n, 2))
        best_sol = np.zeros(n * 3)
        best_sol[0::3] = r_f
        best_sol[1::3] = (centers_f[:, 0] - r_f) / (1.0 - 2.0 * r_f)
        best_sol[2::3] = (centers_f[:, 1] - r_f) / (1.0 - 2.0 * r_f)
        best_val = np.sum(r_f)
        
    # Phase 3: High-precision refinement on the best configuration
    try:
        res_final = minimize(objective, best_sol, method='SLSQP', bounds=bounds,
                             constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'iprint': -1})
        if np.min(constraint_func(res_final.x)) >= -1e-7 and -res_final.fun > best_val:
            best_sol = res_final.x
            best_val = -res_final.fun
    except Exception:
        pass
        
    # Phase 4: Perturb and re-optimize to escape local minima
    np.random.seed(999)
    best_sol_pert = best_sol + np.random.randn(n * 3) * 0.002
    best_sol_pert[0::3] = np.clip(best_sol_pert[0::3], 1e-5, 0.5)
    best_sol_pert[1::3] = np.clip(best_sol_pert[1::3], 0.0, 1.0)
    best_sol_pert[2::3] = np.clip(best_sol_pert[2::3], 0.0, 1.0)
    
    try:
        res_pert = minimize(objective, best_sol_pert, method='SLSQP', bounds=bounds,
                            constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13, 'iprint': -1})
        if np.min(constraint_func(res_pert.x)) >= -1e-7 and -res_pert.fun > best_val:
            best_sol = res_pert.x
            best_val = -res_pert.fun
    except Exception:
        pass
        
    # Reconstruct physical coordinates from parameters
    r_opt = best_sol[0::3]
    u_opt = best_sol[1::3]
    v_opt = best_sol[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    # Final safety checks
    r_opt = np.maximum(r_opt, 1e-9)
    
    # Verify constraints one last time; shrink uniformly if minor numerical drift occurs
    check_vec = np.empty(n * 3)
    check_vec[0::3] = r_opt
    check_vec[1::3] = u_opt
    check_vec[2::3] = v_opt
    
    if np.min(constraint_func(check_vec)) < -1e-9:
        scale = 1.0
        for _ in range(20):
            check_vec[0::3] = r_opt * scale
            if np.min(constraint_func(check_vec)) >= -1e-9:
                break
            scale *= 0.98
        r_opt *= scale
        # Recompute centers with adjusted radii
        x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
        y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
        centers = np.column_stack((x_opt, y_opt))
        
    return centers, r_opt, float(np.sum(r_opt))
