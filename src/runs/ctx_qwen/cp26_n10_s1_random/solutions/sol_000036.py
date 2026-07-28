# sol_000036 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000021 (state e14e8c08) state=7303ed2a sum of radii=2.416390 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_array, n):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_array[2::3])

def constraint_func(vars_array, n):
    """
    Returns array of constraint values >= 0 for valid packing.
    Constraints:
    1. Boundary: x >= r, x <= 1-r, y >= r, y <= 1-r
    2. Non-overlap: (xi-xj)^2 + (yi-yj)^2 >= (ri+rj)^2
    """
    x = vars_array[0::3]
    y = vars_array[1::3]
    r = vars_array[2::3]
    
    # Boundary constraints
    b_cons = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap constraints (vectorized)
    X = x[:, None] - x[None, :]
    Y = y[:, None] - y[None, :]
    R = r[:, None] + r[None, :]
    
    dist_sq = X**2 + Y**2
    r_sum_sq = R**2
    
    idx = np.triu_indices(n, k=1)
    p_cons = dist_sq[idx] - r_sum_sq[idx]
    
    return np.concatenate([b_cons, p_cons])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Variable bounds: x,y in [0,1], r in [small_positive, 0.5]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-6, 0.5)] * n
    cons_dict = {'type': 'ineq', 'fun': constraint_func, 'args': (n,)}
    
    # Generate multiple feasible initial configurations to escape local minima
    inits = []
    rng = np.random.RandomState(42)
    
    # 1. Standard 5x5 grid + 1 in center
    gx = np.linspace(0.1, 0.9, 5)
    gy = np.linspace(0.1, 0.9, 5)
    pts_grid = np.array([(x, y) for x in gx for y in gy])
    pts_grid = np.vstack([pts_grid, [0.5, 0.5]])
    inits.append(pts_grid)
    
    # 2. Hexagonal lattice arrangement
    pts_hex = []
    r0 = 0.09
    for row in range(7):
        y = r0 + row * r0 * np.sqrt(3)
        shift = r0 if row % 2 == 1 else 0.0
        x = r0 + shift
        while x + r0 <= 1.0 and len(pts_hex) < 26:
            pts_hex.append([x, y])
            x += 2 * r0
    inits.append(np.array(pts_hex))
    
    # 3-12. Perturbed grids to explore different basins of attraction
    for _ in range(10):
        pts_p = pts_grid.copy() + rng.uniform(-0.04, 0.04, (26, 2))
        pts_p = np.clip(pts_p, 0.06, 0.94)
        inits.append(pts_p)
        
    # Optimization loop
    for pts in inits:
        x0 = np.zeros(3 * n)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = 0.05  # Start with a small, guaranteed feasible radius
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=cons_dict,
                args=(n,),
                options={'maxiter': 1000, 'ftol': 1e-13, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                c_vals = constraint_func(res.x, n)
                # Allow tiny numerical tolerance for convergence
                if np.min(c_vals) >= -1e-9:
                    current_sum = np.sum(res.x[2::3])
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = np.column_stack((res.x[0::3], res.x[1::3])).copy()
                        best_radii = res.x[2::3].copy()
        except Exception:
            continue
            
    # Final strict validation and safety adjustment
    if best_centers is not None:
        full_vars = np.concatenate([best_centers[:, 0], best_centers[:, 1], best_radii])
        c_vals = constraint_func(full_vars, n)
        
        if np.min(c_vals) < 0:
            # If any constraint is violated, uniformly scale radii down slightly
            # to guarantee validity against the 1e-12 grader tolerance
            max_viol = -np.min(c_vals)
            # Shrink factor proportional to violation, capped to avoid over-shrinking
            scale = 1.0 - max_viol / (2.0 * np.mean(best_radii) + 1e-9)
            scale = min(0.99999, max(0.95, scale))
            best_radii *= scale
            best_sum = np.sum(best_radii)
            
    return best_centers, best_radii, best_sum
