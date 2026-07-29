# sol_000036 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000006 (state 1103014d) state=f7ab721e sum of radii=2.610568 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_vec):
    """Objective function: maximize sum of radii => minimize negative sum."""
    return -np.sum(vars_vec[2::3])

def constraint_func(vars_vec):
    """
    Computes inequality constraints g(vars) >= 0.
    Includes boundary containment and pairwise non-overlap.
    Vectorized for performance.
    """
    n = len(vars_vec) // 3
    x = vars_vec[0::3]
    y = vars_vec[1::3]
    r = vars_vec[2::3]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_boundary = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Broadcasting to compute all pairwise differences
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, None] + r[None, :]
    r_sum_sq = r_sum**2
    
    # Extract upper triangular part (i < j) to avoid duplicates and self-comparison
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c_overlap = dist_sq[mask] - r_sum_sq[mask]
    
    return np.concatenate([c_boundary, c_overlap])

def get_init_vars(n, init_type, seed):
    """Generates a feasible initial configuration based on init_type and seed."""
    np.random.seed(seed)
    vars_vec = np.zeros(3 * n)
    r_start = 0.04  # Small initial radius to ensure strict feasibility
    
    positions = []
    if init_type == 0:
        # Hexagonal lattice packing
        y = r_start
        row = 0
        while len(positions) < n:
            x_off = (row % 2) * r_start
            x = r_start + x_off
            while x <= 1.0 - r_start and len(positions) < n:
                positions.append((x, y))
                x += 2.0 * r_start
            y += np.sqrt(3.0) * r_start
            row += 1
    else:
        # 5x5 Grid + 1 center circle
        for i in range(5):
            for j in range(5):
                positions.append((0.1 + i * 0.2, 0.1 + j * 0.2))
        positions.append((0.5, 0.5))
        
    positions = positions[:n]
    for i, (px, py) in enumerate(positions):
        vars_vec[3*i] = px
        vars_vec[3*i+1] = py
        vars_vec[3*i+2] = r_start
        
    # Add controlled random jitter to escape symmetric local minima
    vars_vec += np.random.uniform(-0.004, 0.004, size=3*n)
    
    # Clip to valid bounds to ensure optimizer starts in a legal region
    vars_vec[0::3] = np.clip(vars_vec[0::3], 0.01, 0.99)
    vars_vec[1::3] = np.clip(vars_vec[1::3], 0.01, 0.99)
    vars_vec[2::3] = np.clip(vars_vec[2::3], 0.01, 0.5)
    
    return vars_vec

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -np.inf
    best_vars = None
    
    # Multiple restarts with diverse initializations
    for seed in range(12):
        init_type = seed % 2  # Alternate between hex and grid
        x0 = get_init_vars(n, init_type, seed)
        
        try:
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
            )
            
            current_sum = -res.fun
            # Evaluate constraint satisfaction strictly
            cons_val = constraint_func(res.x)
            min_cons = np.min(cons_val)
            
            # Accept if constraints are satisfied within numerical tolerance and improves best score
            if min_cons >= -1e-9 and current_sum > best_sum:
                best_sum = current_sum
                best_vars = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization somehow fails completely
    if best_vars is None:
        x0 = get_init_vars(n, 0, 0)
        best_vars = x0
        best_sum = -objective(x0)
        
    # Extract results
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    
    # Ensure non-negative radii (safety against numerical drift)
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
