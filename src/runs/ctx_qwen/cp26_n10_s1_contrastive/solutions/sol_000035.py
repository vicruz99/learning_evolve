# sol_000035 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000006 (state 1103014d) state=cfcb3616 sum of radii=2.625299 correctness=1.0
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
    Constraints: pairwise non-overlap.
    Boundary constraints are handled automatically by parameterization.
    """
    n = 26
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Parameterization maps u, v in [0,1] to x, y such that circle is inside [0,1]^2
    # x = r when u=0 (touching left wall)
    # x = 1-r when u=1 (touching right wall)
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

def run_packing():
    n = 26
    best_sol = None
    best_val = -np.inf
    
    # Bounds: r in [1e-6, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-6, 0.5), (0.0, 1.0), (0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    n_restarts = 25
    
    for seed in range(n_restarts):
        np.random.seed(seed)
        
        # Mix initialization strategies: Hexagonal, Grid, Random
        init_type = seed % 3
        r_init = 0.06 + 0.03 * np.random.rand()
        positions = []
        
        if init_type == 0:  # Hexagonal
            y_pos = r_init
            row = 0
            while len(positions) < n:
                x_pos = r_init + (row % 2) * r_init * 1.05
                while x_pos <= 1.0 - r_init and len(positions) < n:
                    positions.append((x_pos, y_pos))
                    x_pos += 2.0 * r_init * 1.05
                y_pos += np.sqrt(3.0) * r_init * 1.05
                row += 1
        elif init_type == 1:  # Grid
            cols, rows = 5, 6
            step_x = (1.0 - 2.0 * r_init) / (cols - 1)
            step_y = (1.0 - 2.0 * r_init) / (rows - 1)
            for i in range(rows):
                for j in range(cols):
                    if len(positions) < n:
                        positions.append((r_init + j * step_x, r_init + i * step_y))
        else:  # Random within safe margin
            for _ in range(n):
                positions.append((np.random.uniform(r_init, 1.0 - r_init), 
                                 np.random.uniform(r_init, 1.0 - r_init)))
                                 
        positions = positions[:n]
        
        # Convert to normalized u, v coordinates
        denom = 1.0 - 2.0 * r_init
        u_init = np.array([(p[0] - r_init) / denom for p in positions])
        v_init = np.array([(p[1] - r_init) / denom for p in positions])
        
        # Add controlled perturbation
        u_init += np.random.uniform(-0.04, 0.04, n)
        v_init += np.random.uniform(-0.04, 0.04, n)
        u_init = np.clip(u_init, 0.0, 1.0)
        v_init = np.clip(v_init, 0.0, 1.0)
        
        vars0 = np.empty(n * 3)
        vars0[0::3] = r_init
        vars0[1::3] = u_init
        vars0[2::3] = v_init
        
        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'iprint': -1})
            
            # Check constraint satisfaction
            cons_val = constraint_func(res.x)
            if np.min(cons_val) >= -1e-6:
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_sol = res.x
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_sol is None:
        r_f = 0.04
        centers = np.random.rand(n, 2)
        centers = np.clip(centers, r_f, 1.0 - r_f)
        best_sol = np.zeros(n * 3)
        best_sol[0::3] = r_f
        best_sol[1::3] = (centers[:, 0] - r_f) / (1.0 - 2.0 * r_f)
        best_sol[2::3] = (centers[:, 1] - r_f) / (1.0 - 2.0 * r_f)
        best_val = np.sum(r_f)
        
    # Reconstruct centers from optimized parameters
    r_opt = best_sol[0::3]
    u_opt = best_sol[1::3]
    v_opt = best_sol[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    return centers, r_opt, float(np.sum(r_opt))
