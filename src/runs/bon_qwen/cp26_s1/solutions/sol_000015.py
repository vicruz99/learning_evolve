# sol_000015 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 60d0e48a) state=5439dc47 sum of radii=2.606906 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_func(x, n):
    """Objective: maximize sum of radii (minimize negative sum)"""
    return -np.sum(x[2*n:])

def constr_func(x, n):
    """Constraint function: returns array of constraint values >= 0"""
    cx = x[:n]
    cy = x[n:2*n]
    r = x[2*n:]
    
    # Boundary constraints: circles inside [0,1]^2
    cons = [cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r]
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, None] + r[None, :]
    pair_diff = dist_sq - r_sum**2
    
    # Extract upper triangle to avoid duplicates
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    cons.append(pair_diff[mask])
    
    return np.concatenate(cons)

def get_initial_guess(n):
    """Generate initial hexagonal packing configuration"""
    r0 = 0.07
    cx = np.zeros(n)
    cy = np.zeros(n)
    
    # Hexagonal row structure for n=26
    row_counts = [5, 6, 5, 6, 4]
    idx = 0
    y_pos = r0
    
    for r_idx, count in enumerate(row_counts):
        x_shift = r0 if r_idx % 2 == 1 else 0.0
        for c_idx in range(count):
            cx[idx] = r0 + x_shift + c_idx * 2 * r0
            cy[idx] = y_pos
            idx += 1
        y_pos += np.sqrt(3) * r0
        
    # Center the configuration in the unit square
    cx -= cx.mean() - 0.5
    cy -= cy.mean() - 0.5
    
    r = np.full(n, r0)
    return np.concatenate([cx, cy, r])

def run_packing():
    n = 26
    np.random.seed(42)
    
    # Variable bounds: x in [0,1], y in [0,1], r in [1e-6, 0.5]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-6, 0.5)] * n
    constraints = {'type': 'ineq', 'fun': constr_func, 'args': (n,)}
    
    x0 = get_initial_guess(n)
    best_sum = -1.0
    best_x = None
    
    # Multi-start optimization to improve robustness
    for _ in range(5):
        # Perturb initial guess slightly
        x_start = x0 + np.random.randn(3*n) * 0.02
        # Project back to valid bounds
        for i in range(n):
            x_start[i] = np.clip(x_start[i], 0.0, 1.0)
            x_start[n+i] = np.clip(x_start[n+i], 0.0, 1.0)
            x_start[2*n+i] = np.clip(x_start[2*n+i], 1e-6, 0.5)
            
        res = minimize(obj_func, x_start, args=(n,), method='SLSQP',
                       bounds=bounds, constraints=constraints,
                       options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                       
        curr_sum = -res.fun
        # Verify constraints are satisfied within tolerance
        cons_val = constr_func(res.x, n)
        if np.all(cons_val >= -1e-8) and curr_sum > best_sum:
            best_sum = curr_sum
            best_x = res.x.copy()
            
    # Fallback if optimization fails completely
    if best_x is None:
        best_x = x0
        
    # Extract and clean results
    cx = np.clip(best_x[:n], 1e-6, 1.0 - 1e-6)
    cy = np.clip(best_x[n:2*n], 1e-6, 1.0 - 1e-6)
    r = np.clip(best_x[2*n:], 1e-6, 0.5)
    
    centers = np.column_stack((cx, cy))
    return centers, r, np.sum(r)
