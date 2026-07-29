# sol_000081 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000025 (state d15e4e7a) state=849f8f1d sum of radii=2.630439 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
i_idx, j_idx = np.triu_indices(N, k=1)

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[0::3])

def constraint_func(vars_vec):
    """
    Computes inequality constraints: pairwise non-overlap.
    Boundary constraints are automatically satisfied by parameterization.
    Returns array of values that must be >= 0.
    """
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Map normalized [0,1] coordinates to actual positions within [r, 1-r]
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Pairwise squared distances for i < j
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist2 = dx**2 + dy**2
    
    # Required squared distances
    rs = r[i_idx] + r[j_idx]
    
    return dist2 - rs**2

def physical_to_params(pts, r_arr):
    """Converts physical (x,y) points and radii to (r, u, v) parameter space."""
    denom = 1.0 - 2.0 * r_arr
    u = (pts[:, 0] - r_arr) / denom
    v = (pts[:, 1] - r_arr) / denom
    
    # Clip to ensure strict feasibility for optimizer start
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    vars_vec = np.empty(3 * N)
    vars_vec[0::3] = r_arr
    vars_vec[1::3] = u
    vars_vec[2::3] = v
    return vars_vec

def get_init_pts(strategy, seed):
    """Generates initial physical coordinates based on strategy."""
    np.random.seed(seed)
    pts = np.zeros((N, 2))
    
    if strategy == 'hex':
        r_est = 0.09
        y = r_est
        row = 0
        idx = 0
        while y < 1.0 - r_est + 0.01 and idx < N:
            x_off = r_est if row % 2 == 1 else 0.0
            x = r_est + x_off
            while x < 1.0 - r_est + 0.01 and idx < N:
                pts[idx] = [x, y]
                idx += 1
                x += 2.0 * r_est
            y += np.sqrt(3.0) * r_est
            row += 1
    elif strategy == 'grid':
        idx = 0
        for i in range(5):
            for j in range(5):
                pts[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                idx += 1
        pts[25] = [0.5, 0.5]
    elif strategy == 'rand':
        pts = np.random.rand(N, 2)
        pts *= 0.8
        pts += 0.1
    elif strategy == 'corner':
        pts[:4] = [[0.05, 0.05], [0.95, 0.05], [0.05, 0.95], [0.95, 0.95]]
        pts[4:16] = np.random.uniform(0.2, 0.8, (12, 2))
        pts[16:26] = np.random.uniform(0.1, 0.9, (10, 2))
        
    # Add controlled perturbation
    pts += np.random.uniform(-0.02, 0.02, (N, 2))
    pts = np.clip(pts, 0.01, 0.99)
    return pts

def run_packing():
    bounds = [(1e-6, 0.5), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_vars = None
    best_sum = -np.inf
    
    strategies = ['hex', 'grid', 'rand', 'corner']
    
    # Phase 1: Multiple diverse restarts
    for seed in range(40):
        strat = strategies[seed % 4]
        pts = get_init_pts(strat, seed)
        
        # Compute strictly feasible initial radii
        r_arr = np.full(N, 0.08)
        for i in range(N):
            d_wall = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
            d_min = 1.0
            for j in range(N):
                if i != j:
                    d = np.sqrt(np.sum((pts[i]-pts[j])**2))
                    if d < d_min: d_min = d
            r_arr[i] = min(d_wall, d_min * 0.5) * 0.85
            
        x0 = physical_to_params(pts, r_arr)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                c_val = constraint_func(res.x)
                if np.min(c_val) >= -1e-6:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Perturbation refinement to escape local minima
    if best_vars is not None:
        for k in range(30):
            np.random.seed(1000 + k)
            x0 = best_vars.copy()
            
            # Perturb normalized positions and radii
            x0[1::3] += np.random.uniform(-0.06, 0.06, N)
            x0[2::3] += np.random.uniform(-0.06, 0.06, N)
            x0[0::3] *= np.random.uniform(0.92, 1.08, N)
            
            # Enforce bounds
            x0[1::3] = np.clip(x0[1::3], 0.0, 1.0)
            x0[2::3] = np.clip(x0[2::3], 0.0, 1.0)
            x0[0::3] = np.clip(x0[0::3], 1e-6, 0.5)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    c_val = constraint_func(res.x)
                    if np.min(c_val) >= -1e-6:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_vars = res.x.copy()
            except Exception:
                continue
                
    # Phase 3: High-precision polish on best configuration
    if best_vars is not None:
        try:
            res = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if res.success:
                c_val = constraint_func(res.x)
                if np.min(c_val) >= -1e-8:
                    best_vars = res.x
                    best_sum = -res.fun
        except Exception:
            pass
            
    # Fallback valid configuration
    if best_vars is None:
        r_f = 0.05
        u_f = np.random.rand(N)
        v_f = np.random.rand(N)
        best_vars = np.empty(3*N)
        best_vars[0::3] = r_f
        best_vars[1::3] = u_f
        best_vars[2::3] = v_f
        best_sum = float(N * r_f)
        
    # Reconstruct physical centers
    r_opt = best_vars[0::3]
    u_opt = best_vars[1::3]
    v_opt = best_vars[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(best_sum)
