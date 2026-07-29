# sol_000046 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000003 (state f9d5c394) state=190e5aed sum of radii=2.622893 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[0::3])

def compute_constraints(vars_vec):
    """Constraint function: pairwise non-overlap constraints >= 0."""
    radii = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Transform normalized coordinates to actual positions within [r, 1-r]
    x = radii + u * (1.0 - 2.0 * radii)
    y = radii + v * (1.0 - 2.0 * radii)
    
    # Pairwise squared distances
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Required squared distances for non-overlap
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    min_dist_sq = r_sum**2
    
    # Upper triangular indices (i < j) to avoid duplicates and self-constraints
    i_idx, j_idx = np.triu_indices(N_CIRCLES, k=1)
    return dist_sq[i_idx, j_idx] - min_dist_sq[i_idx, j_idx]

def generate_initial_vars(seed, mode='hex'):
    """Generates initial variables for optimization."""
    n = N_CIRCLES
    np.random.seed(seed)
    
    if mode == 'hex':
        # Hexagonal lattice initialization: rows of 6, 5, 6, 5, 4 circles
        # This configuration is known to be near-optimal for N=26
        row_counts = [6, 5, 6, 5, 4]
        r_est = 0.098
        h = r_est * np.sqrt(3.0)
        
        centers = []
        y_pos = r_est
        for r_idx, count in enumerate(row_counts):
            x_start = r_est if r_idx % 2 == 0 else 2.0 * r_est
            for c in range(count):
                centers.append([x_start + c * 2.0 * r_est, y_pos])
            y_pos += h
            
        centers = centers[:n]
        denom = 1.0 - 2.0 * r_est
        u = np.array([(p[0] - r_est) / denom for p in centers])
        v = np.array([(p[1] - r_est) / denom for p in centers])
        
        # Add controlled perturbation to escape symmetry and local minima
        u += np.random.uniform(-0.03, 0.03, n)
        v += np.random.uniform(-0.03, 0.03, n)
        
    else:
        # Random initialization for diversity in topology search
        r_est = 0.06
        u = np.random.uniform(0, 1, n)
        v = np.random.uniform(0, 1, n)
        
    u = np.clip(u, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)
    
    vars0 = np.empty(n * 3)
    vars0[0::3] = r_est
    vars0[1::3] = u
    vars0[2::3] = v
    return vars0

def run_packing():
    n = N_CIRCLES
    best_vars = None
    best_val = -np.inf
    
    # Bounds: r in [1e-6, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-6, 0.5), (0.0, 1.0), (0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    # Multiple restarts to find global optimum
    num_restarts = 25
    for seed in range(num_restarts):
        mode = 'hex' if seed < 15 else 'rand'
        vars0 = generate_initial_vars(seed, mode)
        
        try:
            res = minimize(compute_objective, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'iprint': -1})
            
            val = -res.fun
            # Check constraint satisfaction with tolerance to ensure feasibility
            con_vals = compute_constraints(res.x)
            if np.min(con_vals) >= -1e-5:
                if val > best_val:
                    best_val = val
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    # High-precision refinement on the best configuration found
    if best_vars is not None:
        try:
            res_final = minimize(compute_objective, best_vars, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14, 'iprint': -1})
            best_vars = res_final.x
        except Exception:
            pass
            
    # Fallback to a valid grid packing if optimization fails completely
    if best_vars is None:
        r_fall = 0.05
        u_fall = np.linspace(0.0, 1.0, n)
        v_fall = np.linspace(0.0, 1.0, n)
        best_vars = np.empty(n * 3)
        best_vars[0::3] = r_fall
        best_vars[1::3] = u_fall
        best_vars[2::3] = v_fall
        
    # Reconstruct centers from optimized parameters
    radii = best_vars[0::3]
    u = best_vars[1::3]
    v = best_vars[2::3]
    x = radii + u * (1.0 - 2.0 * radii)
    y = radii + v * (1.0 - 2.0 * radii)
    centers = np.column_stack((x, y))
    
    return centers, radii, float(np.sum(radii))
