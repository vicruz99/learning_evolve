# sol_000063 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000035 (state cfcb3616) state=ec3f197a sum of radii=2.628522 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_vec):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_vec[0::3])

def constraint_func(vars_vec):
    """
    Computes pairwise non-overlap constraints.
    Boundary constraints are automatically satisfied by the parameterization.
    Returns values that must be >= 0.
    """
    n = 26
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Parameterization ensures circles are inside [0,1]^2
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Squared sum of radii
    r_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    # Upper triangular indices for i < j
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Constraint: dist >= r_i + r_j  <=>  dist^2 >= (r_i + r_j)^2
    return dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx]

def compute_feasible_radii(positions):
    """Compute a strictly feasible initial radius for each circle."""
    n = positions.shape[0]
    r = np.zeros(n)
    for i in range(n):
        d_bound = min(positions[i, 0], 1.0 - positions[i, 0],
                      positions[i, 1], 1.0 - positions[i, 1])
        d_min = d_bound
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((positions[i] - positions[j])**2))
                if d < d_min:
                    d_min = d
        # Use a safe fraction to guarantee strict feasibility
        r[i] = 0.35 * d_min
    return r

def run_packing():
    n = 26
    bounds = [(1e-6, 0.5), (0.0, 1.0), (0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sol = None
    best_val = -np.inf
    
    base_positions = []
    
    # 1. Hexagonal lattice initialization
    pts_hex = []
    r_est = 0.1
    y = r_est
    row = 0
    while len(pts_hex) < n:
        x = r_est + (row % 2) * r_est
        while x <= 1.0 - r_est and len(pts_hex) < n:
            pts_hex.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    base_positions.append(np.array(pts_hex))
    
    # 2. Grid initialization
    pts_grid = []
    for i in range(5):
        for j in range(6):
            if len(pts_grid) < n:
                pts_grid.append([0.08 + j*0.16, 0.08 + i*0.15])
    base_positions.append(np.array(pts_grid))
    
    # 3. Random initialization
    np.random.seed(123)
    pts_rand = np.random.uniform(0.1, 0.9, (n, 2))
    base_positions.append(pts_rand)
    
    inits = []
    for base in base_positions:
        for seed in range(12):
            np.random.seed(seed + 100)
            pos = base + np.random.uniform(-0.02, 0.02, (n, 2))
            pos = np.clip(pos, 0.05, 0.95)
            r_init = compute_feasible_radii(pos)
            
            # Map positions to u, v coordinates for the parameterization
            u = np.zeros(n)
            v = np.zeros(n)
            for i in range(n):
                r_i = r_init[i]
                denom = 1.0 - 2.0 * r_i
                if denom <= 1e-9: denom = 1e-9
                u[i] = (pos[i, 0] - r_i) / denom
                v[i] = (pos[i, 1] - r_i) / denom
            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            
            vars0 = np.empty(n * 3)
            vars0[0::3] = r_init
            vars0[1::3] = u
            vars0[2::3] = v
            inits.append(vars0)
            
    # Primary optimization loop
    for vars0 in inits:
        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13, 'iprint': -1})
            cons_val = constraint_func(res.x)
            if np.min(cons_val) >= -1e-7:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_sol = res.x.copy()
        except Exception:
            continue
            
    # Local perturbation refinement to escape local minima
    if best_sol is not None:
        for _ in range(20):
            pert = best_sol.copy()
            pert[0::3] += np.random.uniform(-0.005, 0.005, n)
            pert[0::3] = np.clip(pert[0::3], 1e-6, 0.45)
            pert[1::3] += np.random.uniform(-0.05, 0.05, n)
            pert[2::3] += np.random.uniform(-0.05, 0.05, n)
            pert[1::3] = np.clip(pert[1::3], 0.0, 1.0)
            pert[2::3] = np.clip(pert[2::3], 0.0, 1.0)
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13, 'iprint': -1})
                cons_val = constraint_func(res.x)
                if np.min(cons_val) >= -1e-7:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_sol = res.x.copy()
            except Exception:
                continue

    # Fallback if optimization completely fails
    if best_sol is None:
        best_sol = np.zeros(n * 3)
        best_sol[0::3] = 0.04
        best_sol[1::3] = 0.5
        best_sol[2::3] = 0.5
        best_val = np.sum(best_sol[0::3])
        
    # Reconstruct centers from optimized parameters
    r_opt = best_sol[0::3]
    u_opt = best_sol[1::3]
    v_opt = best_sol[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(best_val)
