# sol_000293 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d1ce3e9) state=ba7c1a2e sum of radii=2.608813 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective function: negative sum of radii (to maximize sum)."""
    return -np.sum(vars[2::3])

def constraint_fun(vars):
    """
    Returns array of constraint values that must be >= 0.
    Constraints: boundaries and non-overlap.
    """
    n = len(vars) // 3
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    vals = list(x - r)
    vals.extend(1.0 - x - r)
    vals.extend(y - r)
    vals.extend(1.0 - y - r)
    
    # Overlap constraints: (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
    # Vectorized computation for speed
    X, Y, R = x[:, None], y[:, None], r[:, None]
    D2 = (X - X.T)**2 + (Y - Y.T)**2
    R_sum = R + R.T
    overlap_vals = D2 - R_sum**2
    
    # Extract strictly upper triangular part
    idx = np.triu_indices(n, k=1)
    vals.extend(overlap_vals[idx])
    
    return np.array(vals)

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Try multiple random seeds to escape local minima
    for seed in [42, 123, 456]:
        np.random.seed(seed)
        xs = np.zeros(n)
        ys = np.zeros(n)
        rs = np.full(n, 0.085)  # Initial radius
        
        # Hexagonal-like lattice initialization
        idx = 0
        row_counts = [5, 6, 5, 5, 5]  # Sums to 26
        y_pos = 0.15
        for i, count in enumerate(row_counts):
            shift = 0.0 if i % 2 == 0 else 0.1
            if count > 1:
                xs[idx:idx+count] = np.linspace(0.1 + shift, 0.9 - shift, count)
            else:
                xs[idx] = 0.5
            ys[idx:idx+count] = y_pos
            idx += count
            y_pos += 0.22
            
        # Add perturbation to break symmetry
        xs += np.random.uniform(-0.01, 0.01, n)
        ys += np.random.uniform(-0.01, 0.01, n)
        xs = np.clip(xs, 0.02, 0.98)
        ys = np.clip(ys, 0.02, 0.98)
        
        # Flatten to optimization variable format: [x0, y0, r0, x1, y1, r1, ...]
        vars0 = np.zeros(3*n)
        vars0[0::3] = xs
        vars0[1::3] = ys
        vars0[2::3] = rs
        
        cons = {'type': 'ineq', 'fun': constraint_fun}
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
        
        # Run SLSQP optimization
        res = minimize(objective, vars0, method='SLSQP', bounds=bounds, constraints=cons,
                       options={'maxiter': 2500, 'ftol': 1e-13})
        
        if res.success:
            current_sum = np.sum(res.x[2::3])
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                best_radii = res.x[2::3]
                
    # Fallback configuration (should not be reached with valid optimization)
    if best_centers is None:
        best_centers = np.column_stack((np.repeat(np.linspace(0.1, 0.9, 5), 6)[:26], 
                                        np.tile(np.linspace(0.1, 0.9, 6), 5)[:26]))
        best_radii = np.full(26, 0.05)
        best_sum = np.sum(best_radii)
        
    # Ensure non-negative radii and return
    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, best_sum
