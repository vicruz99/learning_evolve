# sol_000067 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000035 (state cfcb3616) state=9bfa5850 sum of radii=2.621473 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(vars_vec):
    """Minimize negative sum of radii."""
    return -np.sum(vars_vec[2::3])

def constraint_func(vars_vec):
    """Compute inequality constraints: boundary containment and pairwise separation."""
    x = vars_vec[0::3]
    y = vars_vec[1::3]
    r = vars_vec[2::3]
    
    c_list = []
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c_list.append(x - r)
    c_list.append(1.0 - x - r)
    c_list.append(y - r)
    c_list.append(1.0 - y - r)
    
    # Pairwise separation: dist^2 >= (r_i + r_j)^2
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    c_list.append(dist_sq[I_IDX, J_IDX] - r_sum[I_IDX, J_IDX]**2)
    
    return np.concatenate(c_list)

def generate_init_config(method, seed, perturb_std):
    """Generate a strictly feasible initial center configuration."""
    np.random.seed(seed)
    centers = np.zeros((N, 2))
    
    if method == 'hex':
        r_est = 0.09
        y = r_est
        row = 0
        idx = 0
        while idx < N:
            x = r_est + (row % 2) * r_est * 0.6
            while x <= 1.0 - r_est and idx < N:
                centers[idx] = [x, y]
                idx += 1
                x += 1.6 * r_est
            y += np.sqrt(3.0) * r_est * 0.6
            row += 1
    elif method == 'grid':
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                idx += 1
        if N > 25:
            centers[25] = [0.5, 0.5]
    elif method == 'rand':
        centers = np.random.rand(N, 2)
    elif method == 'corner':
        centers = np.random.rand(N, 2)
        centers[:, 0] = np.clip(centers[:, 0] + np.random.choice([-0.5, 0.5], N), 0.05, 0.95)
        centers[:, 1] = np.clip(centers[:, 1] + np.random.choice([-0.5, 0.5], N), 0.05, 0.95)
        
    if perturb_std > 0:
        centers += np.random.randn(N, 2) * perturb_std
        centers = np.clip(centers, 0.02, 0.98)
        
    return centers

def compute_initial_radii(centers):
    """Compute conservative initial radii guaranteeing strict feasibility."""
    r = np.zeros(N)
    for i in range(N):
        d_wall = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        min_d = 1.0
        for j in range(N):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_d:
                    min_d = d
        r[i] = 0.75 * min(d_wall, min_d / 2.0)
    return np.maximum(r, 1e-5)

def run_packing():
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_vars = None
    best_sum = -np.inf
    
    # Phase 1: Broad search from structured and random initializations
    methods = ['hex', 'grid', 'rand', 'corner']
    for m in methods:
        for s in range(8):
            c_init = generate_init_config(m, s, 0.015)
            r_init = compute_initial_radii(c_init)
            vars0 = np.zeros(3*N)
            vars0[0::3] = c_init[:, 0]
            vars0[1::3] = c_init[:, 1]
            vars0[2::3] = r_init
            
            try:
                res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13})
                # Strict feasibility check
                if np.min(constraint_func(res.x)) >= -1e-7:
                    curr_sum = np.sum(res.x[2::3])
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
            except Exception:
                continue

    # Phase 2: Local perturbation to escape local minima
    if best_vars is not None:
        for k in range(15):
            perturb_amount = 0.008 / (k + 1)
            x_pert = best_vars + np.random.randn(3*N) * perturb_amount
            x_pert[0::3] = np.clip(x_pert[0::3], 0.0, 1.0)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.0, 1.0)
            x_pert[2::3] = np.maximum(x_pert[2::3], 1e-6)
            
            try:
                res_p = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 2500, 'ftol': 1e-13})
                if np.min(constraint_func(res_p.x)) >= -1e-7:
                    curr_sum = np.sum(res_p.x[2::3])
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res_p.x.copy()
            except Exception:
                continue
                
        # Phase 3: High-precision final polish
        try:
            res_f = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                             constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14})
            if np.min(constraint_func(res_f.x)) >= -1e-7:
                best_vars = res_f.x
                best_sum = np.sum(best_vars[2::3])
        except Exception:
            pass

    # Fallback configuration
    if best_vars is None:
        c_fb = generate_init_config('hex', 0, 0.0)
        r_fb = compute_initial_radii(c_fb)
        best_vars = np.zeros(3*N)
        best_vars[0::3] = c_fb[:, 0]
        best_vars[1::3] = c_fb[:, 1]
        best_vars[2::3] = r_fb
        best_sum = np.sum(r_fb)
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    # Safety clamp for numerical drift
    radii = np.maximum(radii, 1e-9)
    return centers, radii, float(np.sum(radii))
