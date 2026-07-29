# sol_000003 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0a5b5ea2) state=f9d5c394 sum of radii=2.612794 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    radii = vars[0::3]
    return -np.sum(radii)

def constraint_func(vars):
    """
    Constraint function: ensures non-overlap between all circle pairs.
    Returns array of constraint values >= 0.
    """
    radii = vars[0::3]
    u = vars[1::3]
    v = vars[2::3]
    n = 26
    
    # Transform normalized coordinates to actual positions
    x = radii + u * (1.0 - 2.0 * radii)
    y = radii + v * (1.0 - 2.0 * radii)
    
    # Compute pairwise squared distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    # Compute squared minimum allowed distances
    r_sum = radii[:, None] + radii[None, :]
    min_dist_sq = r_sum**2
    
    # Extract upper triangular part (i < j) to avoid duplicates and self
    i, j = np.triu_indices(n, k=1)
    return dist_sq[i, j] - min_dist_sq[i, j]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sol = None
    best_val = -np.inf
    
    # Bounds: r in [1e-4, 0.5], u in [0, 1], v in [0, 1]
    bounds = [(1e-4, 0.5), (0, 1), (0, 1)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Run multiple restarts with hexagonal-inspired initialization
    for seed in range(8):
        np.random.seed(seed)
        
        # Hexagonal packing initialization
        r_init = 0.085
        positions = []
        y_pos = r_init
        row = 0
        while len(positions) < n:
            x_pos = r_init
            while x_pos <= 1.0 - r_init:
                if len(positions) < n:
                    positions.append((x_pos, y_pos))
                x_pos += 2.0 * r_init
            y_pos += np.sqrt(3.0) * r_init
            row += 1
            
        # Convert to normalized u, v coordinates
        denom = 1.0 - 2.0 * r_init
        u_init = np.array([(p[0] - r_init) / denom for p in positions])
        v_init = np.array([(p[1] - r_init) / denom for p in positions])
        
        # Add controlled perturbation to escape symmetric local minima
        u_init += np.random.uniform(-0.04, 0.04, n)
        v_init += np.random.uniform(-0.04, 0.04, n)
        u_init = np.clip(u_init, 0.0, 1.0)
        v_init = np.clip(v_init, 0.0, 1.0)
        
        vars0 = np.empty(n * 3)
        vars0[0::3] = r_init
        vars0[1::3] = u_init
        vars0[2::3] = v_init
        
        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
            
            val = -res.fun
            # Accept if significantly better and constraints are satisfied
            if val > best_val and np.min(constraint_func(res.x)) >= -1e-7:
                best_val = val
                best_sol = res.x
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_sol is None:
        r_fall = 0.05
        indices = np.arange(n)
        cols = indices % 6
        rows = indices // 6
        x_fall = 0.1 + cols * 0.18
        y_fall = 0.1 + rows * 0.18
        centers = np.column_stack((x_fall, y_fall))
        radii = np.full(n, r_fall)
        return centers, radii, np.sum(radii)
        
    # Reconstruct centers from optimized parameters
    radii = best_sol[0::3]
    u = best_sol[1::3]
    v = best_sol[2::3]
    x = radii + u * (1.0 - 2.0 * radii)
    y = radii + v * (1.0 - 2.0 * radii)
    centers = np.column_stack((x, y))
    
    return centers, radii, np.sum(radii)
