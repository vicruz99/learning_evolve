# sol_000039 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b444b7b1) state=023a9828 sum of radii=2.502207 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

def get_constraints(vars):
    """
    Computes inequality constraints: fun(vars) >= 0
    Structure: x(26), y(26), r(26)
    """
    x = vars[:26]
    y = vars[26:52]
    r = vars[52:78]
    
    # Boundary constraints
    c_list = [x - r, 1.0 - x - r, y - r, 1.0 - y - r]
    
    # Pairwise non-overlap constraints
    n_pairs = 26 * 25 // 2
    pairwise = np.empty(n_pairs)
    idx = 0
    for i in range(26):
        for j in range(i + 1, 26):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            pairwise[idx] = (dx * dx + dy * dy) - (r[i] + r[j])**2
            idx += 1
            
    c_list.append(pairwise)
    return np.concatenate(c_list)

def objective(vars):
    """Negative sum of radii (to be minimized)"""
    return -np.sum(vars[52:])

def generate_hex_layout(n):
    """Generates an initial hexagonal packing layout"""
    centers = []
    r_init = 0.06
    dx = 2.0 * r_init
    dy = np.sqrt(3) * r_init
    
    y = r_init
    row = 0
    while len(centers) < n:
        x = r_init
        if row % 2 == 1:
            x += r_init  # Shift odd rows for hex pattern
        while x + r_init <= 1.0 and len(centers) < n:
            centers.append([x, y])
            x += dx
        y += dy
        row += 1
    return np.array(centers)

def run_packing():
    np.random.seed(42)
    n = 26
    best_vars = None
    best_obj = np.inf
    
    # Initialize with hex layout
    hex_centers = generate_hex_layout(n)
    init_radii = np.full(n, 0.06)
    
    # Format: x(26), y(26), r(26)
    base_vars = np.concatenate([hex_centers[:, 0], hex_centers[:, 1], init_radii])
    
    bounds = [(0.0, 1.0)] * 52 + [(1e-6, 0.5)] * 26
    constraint = {'type': 'ineq', 'fun': get_constraints}
    
    # Multi-start optimization to escape local minima
    for k in range(4):
        vars0 = base_vars.copy()
        if k > 0:
            # Perturb initial conditions
            vars0 += np.random.uniform(-0.01, 0.01, size=vars0.shape)
            vars0 = np.clip(vars0, 0.0, 1.0)
            vars0[52:] = np.clip(vars0[52:], 1e-6, 0.5)
            
        res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                       constraints=constraint, 
                       options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
        
        # Check success or fallback to best feasible
        is_feasible = np.all(get_constraints(res.x) >= -1e-6)
        if (res.success or is_feasible) and res.fun < best_obj:
            best_obj = res.fun
            best_vars = res.x.copy()
            
    # Extract and format results
    x_best = best_vars[:26]
    y_best = best_vars[26:52]
    r_best = best_vars[52:78]
    
    centers = np.column_stack((x_best, y_best))
    radii = np.maximum(r_best, 0.0)
    
    # Final clipping to strictly satisfy [0,1] bounds
    centers = np.clip(centers, 0.0, 1.0)
    
    return centers, radii, -best_obj
