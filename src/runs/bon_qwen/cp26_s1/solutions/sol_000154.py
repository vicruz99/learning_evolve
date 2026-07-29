# sol_000154 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2bb08abb) state=e900474f sum of radii=2.617481 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Precompute pairwise indices for efficient constraint evaluation
TRI_INDICES = np.triu_indices(26, k=1)

def objective(z):
    """Objective: minimize negative sum of radii"""
    return -np.sum(z[2::3])

def constraints_func(z):
    """Returns array of constraint values. All must be >= 0 for feasibility."""
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Boundary constraints: 4 per circle
    b1 = x - r
    b2 = 1.0 - x - r
    b3 = y - r
    b4 = 1.0 - y - r
    
    # Pairwise non-overlap constraints: distance^2 >= (r1+r2)^2
    i, j = TRI_INDICES
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    dr = r[i] + r[j]
    p_cons = dx**2 + dy**2 - dr**2
    
    return np.concatenate([b1, b2, b3, b4, p_cons])

def get_initial_guess(seed):
    """Generates a feasible initial configuration from a perturbed grid."""
    rng = np.random.RandomState(seed)
    n = 26
    cols, rows = 6, 5
    xs = np.linspace(0.1, 0.9, cols)
    ys = np.linspace(0.1, 0.9, rows)
    
    grid = []
    for y in ys:
        for x in xs:
            grid.append([x, y])
            if len(grid) >= n:
                break
        if len(grid) >= n:
            break
            
    pts = np.array(grid[:n])
    # Add small random jitter to break symmetry
    pts += rng.uniform(-0.03, 0.03, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    
    r_init = np.full(n, 0.04)
    
    z = np.empty(3 * n)
    z[0::3] = pts[:, 0]
    z[1::3] = pts[:, 1]
    z[2::3] = r_init
    return z

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints_func}
    
    best_sum_r = -np.inf
    best_z = None
    
    # Multiple restarts to escape local optima
    for seed in range(20):
        z0 = get_initial_guess(seed)
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-10, 'disp': False})
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum_r:
                    best_sum_r = current_sum
                    best_z = res.x.copy()
        except Exception:
            continue
            
    if best_z is None:
        return np.zeros((n, 2)), np.zeros(n), 0.0
        
    centers = np.column_stack((best_z[0::3], best_z[1::3]))
    radii = best_z[2::3]
    
    # Final feasibility correction: shrink radii slightly if any constraint is violated
    min_con = np.min(constraints_func(best_z))
    if min_con < -1e-9:
        margin = abs(min_con) + 1e-6
        radii = np.maximum(radii - margin, 1e-5)
        best_z[2::3] = radii
        best_sum_r = np.sum(radii)
        
    return centers, radii, best_sum_r
