# sol_000020 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000011 (state bbbe9bd5) state=3986e2e3 sum of radii=2.614250 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    radii = vars[2::3]
    return -np.sum(radii)

def constraint_func(vars):
    """
    Returns a 1D array of constraint values.
    All constraints are formulated as g(vars) >= 0.
    Includes boundary constraints and pairwise non-overlap constraints.
    """
    n = N_CIRCLES
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_boundary = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Non-overlap constraints: dist_sq - (r_i + r_j)^2 >= 0
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    rad_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    # Extract upper triangular indices for i < j
    i, j = np.triu_indices(n, k=1)
    c_overlap = dist_sq[i, j] - rad_sum_sq[i, j]
    
    return np.concatenate([c_boundary, c_overlap])

def get_hexagonal_init(r_init):
    """Generate hexagonal lattice initialization."""
    vars_init = np.zeros(3 * N_CIRCLES)
    idx = 0
    
    y_curr = r_init
    row = 0
    while idx < 3 * N_CIRCLES:
        x_start = r_init if row % 2 == 0 else 2 * r_init
        x_curr = x_start
        while x_curr <= 1.0 - r_init + 1e-9 and idx < 3 * N_CIRCLES:
            vars_init[idx] = x_curr
            vars_init[idx + 1] = y_curr
            vars_init[idx + 2] = r_init
            idx += 3
            x_curr += 2 * r_init
        y_curr += np.sqrt(3) * r_init
        row += 1
    
    return vars_init[:3 * N_CIRCLES]

def get_grid_init():
    """Generate 5x5 grid + 1 circle initialization."""
    vars_init = np.zeros(3 * N_CIRCLES)
    r_init = 0.09
    
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            idx = (i * 5 + j) * 3
            vars_init[idx] = 0.1 + i * 0.2
            vars_init[idx + 1] = 0.1 + j * 0.2
            vars_init[idx + 2] = r_init
    
    # 26th circle in a gap
    idx = 25 * 3
    vars_init[idx] = 0.3
    vars_init[idx + 1] = 0.3
    vars_init[idx + 2] = 0.01
    
    return vars_init

def get_corner_init():
    """Generate corner-focused initialization."""
    vars_init = np.zeros(3 * N_CIRCLES)
    
    # 4 corner circles (larger)
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i, (cx, cy) in enumerate(corners):
        idx = i * 3
        vars_init[idx] = cx
        vars_init[idx + 1] = cy
        vars_init[idx + 2] = 0.12
    
    # Fill remaining with hex pattern
    rest_vars = get_hexagonal_init(0.07)[12:]  # Skip first 4
    vars_init[12:] = rest_vars
    
    return vars_init

def perturb(vars, scale):
    """Add random perturbation to break symmetry."""
    perturbed = vars.copy()
    # Only perturb x, y, not r
    for i in range(0, len(perturbed), 3):
        perturbed[i] += np.random.uniform(-scale, scale)
        perturbed[i + 1] += np.random.uniform(-scale, scale)
    # Clip to valid bounds
    for i in range(0, len(perturbed), 3):
        perturbed[i] = np.clip(perturbed[i], 0.01, 0.99)
        perturbed[i + 1] = np.clip(perturbed[i + 1], 0.01, 0.99)
        perturbed[i + 2] = np.clip(perturbed[i + 2], 0.01, 0.4)
    return perturbed

def run_packing():
    best_vars = None
    best_sum = -1.0
    np.random.seed(42)
    
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
        
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Generate multiple initial configurations
    inits = []
    
    # Hexagonal with different radii
    for r in [0.06, 0.07, 0.08, 0.09]:
        inits.append(get_hexagonal_init(r))
    
    # Grid initialization
    inits.append(get_grid_init())
    
    # Corner-focused
    inits.append(get_corner_init())
    
    # Perturbed versions
    for init in inits[:3]:
        for _ in range(3):
            inits.append(perturb(init, 0.01))
    
    # Local optimization from each init
    for i, vars_init in enumerate(inits):
        try:
            res = minimize(
                objective, 
                vars_init, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
            )
            
            if res.success:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
        except Exception:
            continue
    
    # High-precision refinement on the best configuration found
    if best_vars is not None:
        try:
            res_final = minimize(
                objective, 
                best_vars, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False}
            )
            if res_final.success:
                best_vars = res_final.x
        except Exception:
            pass
    
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    
    # Ensure non-negative radii and validity
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
