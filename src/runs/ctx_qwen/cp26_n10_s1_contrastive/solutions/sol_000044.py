# sol_000044 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000003 (state f9d5c394) state=a8ad9cca sum of radii=2.602430 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraint_func(vars):
    """
    Computes inequality constraints g(vars) >= 0.
    Includes boundary containment and pairwise non-overlap.
    """
    n = 26
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    c_list = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c_list.append(x - r)
    c_list.append(1.0 - x - r)
    c_list.append(y - r)
    c_list.append(1.0 - y - r)
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    r_sum = r[i_idx] + r[j_idx]
    c_list.append(dx*dx + dy*dy - r_sum*r_sum)
    
    return np.concatenate(c_list)

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -np.inf
    best_vars = None
    
    # Multi-start optimization with diverse initializations
    for seed in range(40):
        np.random.seed(seed)
        vars0 = np.zeros(3 * n)
        
        if seed < 20:
            # Hexagonal lattice initialization with perturbation
            r0 = 0.05
            positions = []
            y_pos = r0
            row = 0
            while len(positions) < n:
                x_pos = r0 + (row % 2) * r0
                while x_pos <= 1.0 - r0:
                    positions.append((x_pos, y_pos))
                    x_pos += 2.0 * r0
                y_pos += np.sqrt(3.0) * r0
                row += 1
            positions = positions[:n]
            
            for i in range(n):
                vars0[3*i] = positions[i][0] + np.random.uniform(-0.03, 0.03)
                vars0[3*i+1] = positions[i][1] + np.random.uniform(-0.03, 0.03)
                vars0[3*i+2] = r0 + np.random.uniform(-0.01, 0.01)
        else:
            # Random spread initialization
            vars0[0::3] = np.random.uniform(0.1, 0.9, n)
            vars0[1::3] = np.random.uniform(0.1, 0.9, n)
            vars0[2::3] = np.random.uniform(0.03, 0.09, n)
            
        # Ensure initial bounds are respected
        vars0[0::3] = np.clip(vars0[0::3], 0.02, 0.98)
        vars0[1::3] = np.clip(vars0[1::3], 0.02, 0.98)
        vars0[2::3] = np.clip(vars0[2::3], 0.02, 0.4)
        
        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-11})
            
            if res.success:
                current_sum = -res.fun
                # Verify constraints are satisfied within numerical tolerance
                if np.min(constraint_func(res.x)) >= -1e-7:
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Final high-precision refinement on the best configuration found
    if best_vars is not None:
        try:
            res_final = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14})
            if res_final.success and np.min(constraint_func(res_final.x)) >= -1e-9:
                best_vars = res_final.x
        except Exception:
            pass
    else:
        # Fallback to a strictly valid small packing
        np.random.seed(42)
        centers_fall = np.random.uniform(0.1, 0.9, (n, 2))
        radii_fall = np.full(n, 0.04)
        best_vars = np.zeros(3*n)
        best_vars[0::3] = centers_fall[:, 0]
        best_vars[1::3] = centers_fall[:, 1]
        best_vars[2::3] = radii_fall
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    
    return centers, radii, float(np.sum(radii))
