# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000011 (state bbbe9bd5) state=6c6a55f2 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    radii = vars.reshape(N_CIRCLES, 3)[:, 2]
    return -np.sum(radii)

def constraint_func(vars):
    """
    Returns a 1D array of constraint values.
    All constraints are formulated as g(vars) >= 0.
    Includes boundary constraints and pairwise non-overlap constraints.
    """
    pts = vars.reshape(N_CIRCLES, 3)
    centers = pts[:, :2]
    radii = pts[:, 2]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_boundary = np.column_stack([
        centers[:, 0] - radii,
        1.0 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1.0 - centers[:, 1] - radii
    ]).flatten()
    
    # Non-overlap constraints: dist_sq - (r_i + r_j)^2 >= 0
    dx = centers[:, np.newaxis, 0] - centers[np.newaxis, :, 0]
    dy = centers[:, np.newaxis, 1] - centers[np.newaxis, :, 1]
    dist_sq = dx**2 + dy**2
    rad_sum_sq = (radii[:, np.newaxis] + radii[np.newaxis, :])**2
    
    # Extract upper triangular indices for i < j
    i, j = np.triu_indices(N_CIRCLES, k=1)
    c_overlap = dist_sq[i, j] - rad_sum_sq[i, j]
    
    return np.concatenate([c_boundary, c_overlap])

def run_packing():
    np.random.seed(123)
    best_vars = None
    best_sum = -1.0
    
    # Define bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    hex_pts = []
    r_h = 0.085
    y_curr = r_h
    row = 0
    while len(hex_pts) < N_CIRCLES:
        x_start = r_h + (row % 2) * r_h
        x = x_start
        while x <= 1.0 - r_h and len(hex_pts) < N_CIRCLES:
            hex_pts.append([x, y_curr])
            x += 2 * r_h
        y_curr += np.sqrt(3) * r_h
        row += 1
    inits.append(np.array(hex_pts[:N_CIRCLES]))
    
    # 2. 5x5 grid + 1 in center
    grid_pts = np.array([[0.1 + 0.2*i, 0.1 + 0.2*j] for i in range(5) for j in range(5)] + [[0.5, 0.5]])
    inits.append(grid_pts)
    
    # 3. Perturbed Hexagonal
    inits.append(hex_pts + np.random.randn(len(hex_pts), 2) * 0.015)
    
    # 4. Perturbed Grid
    inits.append(grid_pts + np.random.randn(26, 2) * 0.015)
    
    # 5-10. Random perturbations of base configs
    for _ in range(6):
        base = inits[np.random.randint(2)]
        pert = base + np.random.randn(N_CIRCLES, 2) * 0.02
        inits.append(pert)
        
    for idx, centers_init in enumerate(inits):
        # Clip centers to valid range
        centers_init = np.clip(centers_init, 0.02, 0.98)
        
        # Initialize radii conservatively to ensure starting point is feasible
        r_init = 0.04 + 0.02 * np.random.rand()
        vars_init = np.zeros(3 * N_CIRCLES)
        for k in range(N_CIRCLES):
            vars_init[3*k] = centers_init[k, 0]
            vars_init[3*k+1] = centers_init[k, 1]
            
            # Ensure initial radius fits strictly inside boundaries
            max_r_bound = min(centers_init[k, 0], 1.0 - centers_init[k, 0], 
                              centers_init[k, 1], 1.0 - centers_init[k, 1])
            vars_init[3*k+2] = min(r_init, max_r_bound * 0.85)
            
        # Run optimization
        try:
            res = minimize(objective, vars_init, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'iprint': -1})
            
            if res.success:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x
        except Exception:
            continue
            
    if best_vars is None:
        # Fallback to a valid small packing
        centers = grid_pts
        radii = np.full(N_CIRCLES, 0.09)
        return centers, radii, np.sum(radii)
        
    centers = best_vars.reshape(N_CIRCLES, 3)[:, :2]
    radii = best_vars.reshape(N_CIRCLES, 3)[:, 2]
    
    # Safety: ensure non-negative radii
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, np.sum(radii)
