# sol_000110 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a98c42c6) state=40d46665 sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

# Set fixed seed for reproducibility in initialization
np.random.seed(42)

def _get_vars(params, n):
    """Helper to unpack the flat parameter array."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    return x, y, r

def _objective(params, n):
    """Objective: maximize sum of radii."""
    _, _, r = _get_vars(params, n)
    return -np.sum(r)

def _constraints_func(params, n):
    """
    Vectorized function for all inequality constraints >= 0.
    Returns an array of length 429 (104 boundary + 325 pairwise).
    """
    x, y, r = _get_vars(params, n)
    out = np.empty(n * 4 + n * (n - 1) // 2)
    
    # Boundary constraints (104)
    # x - r >= 0
    out[0::4] = x - r
    # 1 - (x + r) >= 0
    out[1::4] = 1 - (x + r)
    # y - r >= 0
    out[2::4] = y - r
    # 1 - (y + r) >= 0
    out[3::4] = 1 - (y + r)
    
    # Pairwise constraints (325)
    # d^2 - (r_i + r_j)^2 >= 0
    idx = n * 4
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist_sq = dx*dx + dy*dy
            r_sum_sq = (r[i] + r[j])**2
            out[idx] = dist_sq - r_sum_sq
            idx += 1
    return out

def _constraints_jac(params, n):
    """
    Vectorized Jacobian of constraints.
    Returns a matrix of shape (429, 78).
    Rows correspond to constraints, columns to variables.
    """
    x, y, r = _get_vars(params, n)
    n_constraints = n * 4 + n * (n - 1) // 2
    jac = np.zeros((n_constraints, 3 * n))
    
    # Boundary Jacobian
    # x - r >= 0 (indices 0, 4, 8...)
    jac[0::4, 0::3] = 1.0
    jac[0::4, 2::3] = -1.0
    
    # 1 - (x + r) >= 0 (indices 1, 5, 9...)
    jac[1::4, 0::3] = -1.0
    jac[1::4, 2::3] = -1.0
    
    # y - r >= 0 (indices 2, 6, 10...)
    jac[2::4, 1::3] = 1.0
    jac[2::4, 2::3] = -1.0
    
    # 1 - (y + r) >= 0 (indices 3, 7, 11...)
    jac[3::4, 1::3] = -1.0
    jac[3::4, 2::3] = -1.0
    
    # Pairwise Jacobian
    idx = n * 4
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            r_sum = r[i] + r[j]
            
            # d^2 - (r_i + r_j)^2
            # Derivatives w.r.t x_i, x_j
            jac[idx, 3*i] = 2 * dx
            jac[idx, 3*j] = -2 * dx
            
            # Derivatives w.r.t y_i, y_j
            jac[idx, 3*i + 1] = 2 * dy
            jac[idx, 3*j + 1] = -2 * dy
            
            # Derivatives w.r.t r_i, r_j
            jac[idx, 3*i + 2] = -2 * r_sum
            jac[idx, 3*j + 2] = -2 * r_sum
            
            idx += 1
    return jac

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    dim = 3 * n
    
    # Initialize centers on a dense hexagonal grid
    centers = []
    # Hex grid spacing approx 0.18
    d = 0.18
    h = d * np.sqrt(3) / 2
    y_curr = 0.15
    row = 0
    while len(centers) < n:
        x_curr = 0.15 if row % 2 == 0 else 0.15 + d/2
        while x_curr < 0.85 and len(centers) < n:
            centers.append([x_curr, y_curr])
            x_curr += d
        y_curr += h
        row += 1
        
    # Add jitter to initialization to avoid perfect grid symmetry which can be a local trap
    centers = np.array(centers)
    centers += np.random.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.02, 0.98)
    
    best_sum_r = -np.inf
    best_x = None
    
    # Run multiple optimizations with different random seeds/perturbations
    # to escape local optima
    for attempt in range(10):
        np.random.seed(attempt * 123)
        
        # Re-jitter centers for this attempt
        current_centers = centers.copy()
        current_centers += np.random.normal(0, 0.01, centers.shape)
        current_centers = np.clip(current_centers, 0.02, 0.98)
        
        # Initial radii small to ensure feasibility
        init_r = np.full(n, 0.02)
        
        params0 = np.zeros(dim)
        params0[0::3] = current_centers[:, 0]
        params0[1::3] = current_centers[:, 1]
        params0[2::3] = init_r
        
        bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
        
        nl_cons = NonlinearConstraint(
            fun=lambda p: _constraints_func(p, n),
            lb=0,
            ub=np.inf,
            jac=lambda p: _constraints_jac(p, n)
        )
        
        try:
            res = minimize(
                _objective,
                params0,
                args=(n,),
                method='trust-constr',
                bounds=bounds,
                constraints=nl_cons,
                options={'maxiter': 500, 'verbose': 0}
            )
            
            if res.success and -res.fun > best_sum_r:
                best_sum_r = -res.fun
                best_x = res.x.copy()
        except Exception:
            continue

    # Final extraction and validation check
    if best_x is None:
        # Fallback if optimization failed
        best_x = np.zeros(dim)
        best_x[0::3] = current_centers[:, 0]
        best_x[1::3] = current_centers[:, 1]
        best_x[2::3] = 0.01
        best_sum_r = 0.26

    final_centers = np.column_stack((best_x[0::3], best_x[1::3]))
    final_radii = best_x[2::3]
    
    # Ensure non-negative radii (solver might dip slightly below 0 in numerical noise)
    final_radii = np.maximum(final_radii, 1e-9)
    
    # Re-calculate sum to be precise
    actual_sum = np.sum(final_radii)
    
    return final_centers, final_radii, float(actual_sum)
