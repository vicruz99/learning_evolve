# sol_000373 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 469c683e) state=fc5605c2 sum of radii=2.570843 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_initial_guess(n=26):
    """Generate an initial hexagonal lattice arrangement scaled to the unit square."""
    positions = []
    row = 0
    while len(positions) < n:
        col = 0
        while len(positions) < n:
            x = col * 1.0 + (0.5 if row % 2 == 1 else 0.0)
            y = row * np.sqrt(3) / 2
            positions.append((x, y))
            col += 1
        row += 1
        
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    
    # Normalize to [0, 1]
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)
    
    # Shrink slightly to leave room for optimization expansion
    margin = 0.15
    xs = margin + (1 - 2 * margin) * xs
    ys = margin + (1 - 2 * margin) * ys
    
    vars0 = np.zeros(3 * n)
    vars0[0::3] = xs
    vars0[1::3] = ys
    vars0[2::3] = 0.10  # Initial guess for radii
    return vars0

def objective(vars):
    """Negative sum of radii (to be minimized)."""
    return -np.sum(vars[2::3])

def constraint_func(vars):
    """Returns a 1D array of all inequality constraint values."""
    n = 26
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c1 = x - r
    c2 = 1.0 - x - r
    c3 = y - r
    c4 = 1.0 - y - r
    
    # Separation constraints: ||c_i - c_j|| >= r_i + r_j
    # Compute pairwise distances efficiently
    xi = x[:, None]
    yi = y[:, None]
    ri = r[:, None]
    
    dx = xi - xi.T
    dy = yi - yi.T
    # Add small epsilon to prevent sqrt(0) singularities
    dist = np.sqrt(dx**2 + dy**2 + 1e-16)
    
    sep = dist - (ri + ri.T)
    
    # Extract upper triangular part (i < j) to avoid duplicates
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    sep_flat = sep[mask]
    
    return np.concatenate([c1, c2, c3, c4, sep_flat])

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Define bounds for variables: x,y in [0,1], r in [0,1]
    bounds = [(0, 1)] * (2 * n) + [(0, 1)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Run optimization with multiple randomized restarts to escape local minima
    for seed in range(8):
        np.random.seed(seed)
        vars0 = get_initial_guess(n)
        
        # Add controlled noise to initial positions
        noise = np.random.randn(len(vars0)) * 0.015
        vars0 += noise
        
        # Clip to ensure feasible starting point
        vars0[0::3] = np.clip(vars0[0::3], 0.05, 0.95)
        vars0[1::3] = np.clip(vars0[1::3], 0.05, 0.95)
        vars0[2::3] = np.clip(vars0[2::3], 0.01, 0.30)
        
        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            
            cur_sum = np.sum(res.x[2::3])
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                best_radii = res.x[2::3]
        except Exception:
            continue
            
    # Fallback if optimization fails completely
    if best_radii is None:
        vars0 = get_initial_guess(n)
        best_centers = np.column_stack((vars0[0::3], vars0[1::3]))
        best_radii = vars0[2::3]
        best_sum = np.sum(best_radii)
        
    # Ensure strict non-negativity
    best_radii = np.maximum(best_radii, 1e-9)
    
    return best_centers, best_radii, best_sum
