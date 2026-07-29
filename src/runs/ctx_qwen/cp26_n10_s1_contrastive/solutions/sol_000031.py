# sol_000031 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000006 (state 1103014d) state=906e514b sum of radii=2.621793 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    n = len(vars_vec) // 3
    return -np.sum(vars_vec[0::3])

def constraints(vars_vec):
    """
    Returns inequality constraints >= 0.
    Uses parameterization x = r + u*(1-2r), y = r + v*(1-2r) to automatically
    satisfy boundary constraints. Only pairwise non-overlap constraints remain.
    """
    n = len(vars_vec) // 3
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Transform normalized coordinates to actual positions
    factor = 1.0 - 2.0 * r
    x = r + u * factor
    y = r + v * factor
    
    # Pairwise non-overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, None] + r[None, :]
    r_sum_sq = r_sum**2
    
    # Extract upper triangular part to avoid duplicates and self-comparison
    i, j = np.triu_indices(n, k=1)
    return dist_sq[i, j] - r_sum_sq[i, j]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -np.inf
    best_vars = None
    
    # Bounds: r in [1e-4, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-4, 0.5), (0.0, 1.0), (0.0, 1.0)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    initial_configs = []
    
    # Config 1: Hexagonal layout (rows: 6, 5, 6, 5, 4)
    rows = [6, 5, 6, 5, 4]
    r_init = 0.055  # Safe initial radius to guarantee strict feasibility
    y_c = 0.15
    cx, cy = [], []
    for row_idx, count in enumerate(rows):
        x_c = 0.15 + (0.5 if row_idx % 2 == 1 else 0.0) * 0.16
        for _ in range(count):
            cx.append(x_c)
            cy.append(y_c)
            x_c += 0.16
        y_c += 0.14
        
    cx = np.array(cx[:n])
    cy = np.array(cy[:n])
    
    u1 = (cx - r_init) / (1.0 - 2.0 * r_init)
    v1 = (cy - r_init) / (1.0 - 2.0 * r_init)
    u1 = np.clip(u1, 0.0, 1.0)
    v1 = np.clip(v1, 0.0, 1.0)
    
    vars1 = np.empty(3 * n)
    vars1[0::3] = r_init
    vars1[1::3] = u1
    vars1[2::3] = v1
    initial_configs.append(vars1)
    
    # Config 2: 5x5 grid + 1 center
    cx2, cy2 = [], []
    for i in range(5):
        for j in range(5):
            cx2.append(0.1 + i * 0.2)
            cy2.append(0.1 + j * 0.2)
    cx2.append(0.5)
    cy2.append(0.5)
    cx2 = np.array(cx2[:n])
    cy2 = np.array(cy2[:n])
    
    u2 = (cx2 - r_init) / (1.0 - 2.0 * r_init)
    v2 = (cy2 - r_init) / (1.0 - 2.0 * r_init)
    u2 = np.clip(u2, 0.0, 1.0)
    v2 = np.clip(v2, 0.0, 1.0)
    
    vars2 = np.empty(3 * n)
    vars2[0::3] = r_init
    vars2[1::3] = u2
    vars2[2::3] = v2
    initial_configs.append(vars2)
    
    # Optimization with multiple restarts
    for base_init in initial_configs:
        for seed in range(8):
            np.random.seed(seed)
            # Perturb initial guess to escape symmetric local minima
            curr = base_init + np.random.uniform(-0.04, 0.04, 3 * n)
            curr[0::3] = np.clip(curr[0::3], 1e-4, 0.45)
            curr[1::3] = np.clip(curr[1::3], 0.0, 1.0)
            curr[2::3] = np.clip(curr[2::3], 0.0, 1.0)
            
            try:
                res = minimize(
                    objective, 
                    curr, 
                    method='SLSQP', 
                    bounds=bounds,
                    constraints=cons, 
                    options={'maxiter': 4000, 'ftol': 1e-11}
                )
                
                if res.success:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
            except Exception:
                pass
                
    # Fallback if optimization fails completely
    if best_vars is None:
        best_vars = initial_configs[0]
        
    # Reconstruct centers and radii
    r = best_vars[0::3]
    u = best_vars[1::3]
    v = best_vars[2::3]
    factor = 1.0 - 2.0 * r
    x = r + u * factor
    y = r + v * factor
    
    centers = np.column_stack((x, y))
    radii = r
    return centers, radii, float(np.sum(radii))
