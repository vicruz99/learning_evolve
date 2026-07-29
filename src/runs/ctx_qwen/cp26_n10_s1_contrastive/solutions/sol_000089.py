# sol_000089 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000042 (state 26164787) state=c83c6c93 sum of radii=2.630624 correctness=1.0
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
    Computes inequality constraints g(vars_vec) >= 0.
    Uses parameterization to automatically satisfy boundary constraints.
    Only pairwise non-overlap constraints are enforced.
    """
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Parameterization maps u, v in [0,1] to x, y such that circle is strictly inside [0,1]^2
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Squared sum of radii
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    # Constraint: dist^2 >= (r_i + r_j)^2
    return dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx]

def generate_init(seed, base_r):
    """Generates a strictly feasible initial configuration based on a hexagonal lattice."""
    np.random.seed(seed)
    pts = []
    y_pos = base_r
    row = 0
    while len(pts) < N:
        x_off = (row % 2) * base_r
        x_pos = base_r + x_off
        while x_pos <= 1.0 - base_r and len(pts) < N:
            pts.append([x_pos, y_pos])
            x_pos += 2.0 * base_r
        y_pos += np.sqrt(3.0) * base_r
        row += 1
        
    centers = np.array(pts[:N])
    
    # Add controlled perturbation to break symmetry
    centers += np.random.uniform(-0.02, 0.02, (N, 2))
    centers = np.clip(centers, 0.02, 0.98)
    
    # Compute strictly feasible initial radii
    r_vals = np.zeros(N)
    for i in range(N):
        d_bound = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        d_pair = 1.0
        for j in range(N):
            if i != j:
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d < d_pair:
                    d_pair = d
        r_vals[i] = 0.4 * min(d_bound, d_pair / 2.0)
        
    r_init = r_vals.copy()
    denom = np.clip(1.0 - 2.0 * r_init, 1e-6, 1.0)
    u_init = np.clip((centers[:, 0] - r_init) / denom, 0.0, 1.0)
    v_init = np.clip((centers[:, 1] - r_init) / denom, 0.0, 1.0)
    
    vars0 = np.empty(3 * N)
    vars0[0::3] = r_init
    vars0[1::3] = u_init
    vars0[2::3] = v_init
    return vars0

def run_packing():
    """
    Solves the circle packing problem for N=26 in a unit square.
    Returns (centers, radii, sum_radii).
    """
    bounds = [(1e-6, 0.5), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_vars = None
    best_sum = -1.0
    
    # Phase 1: Broad search from diverse hexagonal initializations
    for seed in range(50):
        base_r = 0.05 + 0.03 * (seed % 4)
        try:
            x0 = generate_init(seed, base_r)
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13, 'disp': False})
            
            cons_val = constraint_func(res.x)
            if np.min(cons_val) >= -1e-7:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_vars is not None:
        for k in range(30):
            x0 = best_vars.copy()
            pert = np.random.randn(3 * N)
            # Perturb positions more aggressively than radii to explore topology changes
            x0[1::3] += pert[1::3] * 0.06
            x0[2::3] += pert[2::3] * 0.06
            x0[0::3] += pert[0::3] * 0.005
            
            x0[0::3] = np.clip(x0[0::3], 1e-6, 0.5)
            x0[1::3] = np.clip(x0[1::3], 0.0, 1.0)
            x0[2::3] = np.clip(x0[2::3], 0.0, 1.0)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13, 'disp': False})
                cons_val = constraint_func(res.x)
                if np.min(cons_val) >= -1e-7:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
            except Exception:
                continue
                
    # Phase 3: High-precision polynomial-time polish
    if best_vars is not None:
        try:
            res = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if np.min(constraint_func(res.x)) >= -1e-7:
                best_vars = res.x
                best_sum = -res.fun
        except Exception:
            pass
            
    # Fallback configuration (should not be reached given robust initialization)
    if best_vars is None:
        centers_f = np.zeros((N, 2))
        idx = 0
        for i in range(5):
            for j in range(5):
                centers_f[idx] = [0.1 + 0.2 * i, 0.1 + 0.2 * j]
                idx += 1
        centers_f[25] = [0.5, 0.5]
        r_f = np.full(N, 0.09)
        
        best_vars = np.empty(3 * N)
        best_vars[0::3] = r_f
        best_vars[1::3] = (centers_f[:, 0] - r_f) / (1.0 - 2.0 * r_f)
        best_vars[2::3] = (centers_f[:, 1] - r_f) / (1.0 - 2.0 * r_f)
        best_sum = np.sum(r_f)

    # Reconstruct centers from optimized parameters
    r_opt = best_vars[0::3]
    u_opt = best_vars[1::3]
    v_opt = best_vars[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(best_sum)
