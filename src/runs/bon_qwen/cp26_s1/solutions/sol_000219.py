# sol_000219 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state df9a626f) state=03959c52 sum of radii=2.626678 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

def obj_func(vars, n):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(vars[2::3])

def nonlin_constraints(vars, n):
    """
    Computes non-linear overlap constraints.
    Returns an array of values that must be <= 0.
    Constraint: (r_i + r_j)^2 - ((x_i - x_j)^2 + (y_i - y_j)^2) <= 0
    """
    pts = vars.reshape(n, 3)
    X, Y, R = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Vectorized difference calculations
    dx = X[:, None] - X[None, :]
    dy = Y[:, None] - Y[None, :]
    rs = R[:, None] + R[None, :]
    
    dist_sq = dx**2 + dy**2
    vals = rs**2 - dist_sq
    
    # We only need constraints for i < j (upper triangle)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return vals[mask]

def nonlin_constraints_wrapper(vars):
    """Wrapper to pass to NonlinearConstraint without lambda"""
    n = vars.shape[0] // 3
    return nonlin_constraints(vars, n)

def run_packing():
    n = 26
    num_vars = 3 * n
    
    # Setup bounds: x, y in [0, 1], r in [1e-6, 0.5]
    bounds = []
    for i in range(num_vars):
        if i % 3 == 2:  # radius
            bounds.append((1e-6, 0.5))
        else:  # x, y
            bounds.append((0.0, 1.0))
            
    # Setup linear constraints for boundaries: A @ vars >= lb
    # Constraints: x_i - r_i >= 0, -x_i - r_i >= -1, y_i - r_i >= 0, -y_i - r_i >= -1
    A_lin = np.zeros((4 * n, num_vars))
    lb_lin = np.zeros(4 * n)
    
    for i in range(n):
        idx = 3 * i
        # x_i - r_i >= 0
        A_lin[4*i, idx] = 1.0
        A_lin[4*i, idx+2] = -1.0
        lb_lin[4*i] = 0.0
        
        # -x_i - r_i >= -1  <=> x_i + r_i <= 1
        A_lin[4*i+1, idx] = -1.0
        A_lin[4*i+1, idx+2] = -1.0
        lb_lin[4*i+1] = -1.0
        
        # y_i - r_i >= 0
        A_lin[4*i+2, idx+1] = 1.0
        A_lin[4*i+2, idx+2] = -1.0
        lb_lin[4*i+2] = 0.0
        
        # -y_i - r_i >= -1 <=> y_i + r_i <= 1
        A_lin[4*i+3, idx+1] = -1.0
        A_lin[4*i+3, idx+2] = -1.0
        lb_lin[4*i+3] = -1.0
        
    lin_cons = LinearConstraint(A_lin, lb_lin, np.inf)
    
    # Setup non-linear constraints for overlaps: g(vars) <= 0
    nl_cons = NonlinearConstraint(nonlin_constraints_wrapper, -np.inf, 0.0)
    
    best_res = None
    best_sum_r = -1.0
    rng = np.random.default_rng(42)
    
    # Try multiple random starts to avoid local minima
    for trial in range(5):
        x0 = np.zeros(num_vars)
        for i in range(n):
            # Hexagonal-ish grid initialization
            row = i // 5
            col = i % 5
            base_x = 0.2 + col * 0.15 + (row % 2) * 0.075
            base_y = 0.2 + row * 0.15
            
            x0[3*i] = base_x
            x0[3*i+1] = base_y
            x0[3*i+2] = 0.08  # Initial small radius
            
            # Add perturbation for diversity across trials
            if trial > 0:
                x0[3*i] += rng.uniform(-0.05, 0.05)
                x0[3*i+1] += rng.uniform(-0.05, 0.05)
                
        # Clip initial guess to bounds
        for i in range(num_vars):
            lo, hi = bounds[i]
            x0[i] = np.clip(x0[i], lo, hi)
            
        try:
            res = minimize(obj_func, x0, args=(n,), 
                           method='SLSQP', bounds=bounds, 
                           constraints=[lin_cons, nl_cons],
                           options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False})
            
            if res.success:
                sum_r = -res.fun
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_res = res
        except Exception:
            continue
            
    if best_res is None:
        # Fallback to empty valid packing if optimization fails
        return np.zeros((n, 2)), np.zeros(n), 0.0
        
    vars_opt = best_res.x
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = vars_opt[3*i]
        centers[i, 1] = vars_opt[3*i+1]
        radii[i] = vars_opt[3*i+2]
        
    return centers, radii, np.sum(radii)
