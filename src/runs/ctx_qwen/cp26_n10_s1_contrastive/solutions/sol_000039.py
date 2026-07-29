# sol_000039 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000003 (state f9d5c394) state=92a7236a sum of radii=2.615367 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective_func(vars_vec):
    """Minimize negative sum of radii (equivalent to maximizing sum of radii)."""
    return -np.sum(vars_vec[0::3])

def constraint_func(vars_vec):
    """
    Returns pairwise non-overlap constraints.
    Implicitly assumes boundary constraints are satisfied by the x = r + u*(1-2r) transformation.
    """
    radii = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    n = 26
    
    # Transform normalized coordinates to actual positions within [r, 1-r]
    x = radii + u * (1.0 - 2.0 * radii)
    y = radii + v * (1.0 - 2.0 * radii)
    
    # Vectorized pairwise squared distances
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    # Vectorized squared minimum allowed distances (ri + rj)^2
    r_sum = radii[:, None] + radii[None, :]
    min_dist_sq = r_sum**2
    
    # Extract upper triangular part (i < j) to avoid duplicates and self-comparisons
    i, j = np.triu_indices(n, k=1)
    return dist_sq[i, j] - min_dist_sq[i, j]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(1e-5, 0.5), (0, 1), (0, 1)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_val = -np.inf
    best_sol = None
    
    # Phase 1: Global search with multiple diverse restarts
    for seed in range(30):
        np.random.seed(seed)
        
        r_init = 0.085 + np.random.uniform(-0.005, 0.01)
        positions = []
        
        # Hexagonal packing initialization
        y_pos = r_init
        row = 0
        while len(positions) < n:
            x_pos = r_init
            while x_pos <= 1.0 - r_init and len(positions) < n:
                positions.append((x_pos, y_pos))
                x_pos += 2.0 * r_init
            y_pos += np.sqrt(3.0) * r_init
            row += 1
            
        positions = positions[:n]
        denom = np.maximum(1.0 - 2.0 * r_init, 1e-9)
        u_init = np.array([(p[0] - r_init) / denom for p in positions])
        v_init = np.array([(p[1] - r_init) / denom for p in positions])
        
        # Perturb to break symmetry and explore different basins of attraction
        u_init += np.random.uniform(-0.1, 0.1, n)
        v_init += np.random.uniform(-0.1, 0.1, n)
        u_init = np.clip(u_init, 0.0, 1.0)
        v_init = np.clip(v_init, 0.0, 1.0)
        
        vars0 = np.empty(n * 3)
        vars0[0::3] = r_init
        vars0[1::3] = u_init
        vars0[2::3] = v_init
        
        try:
            res = minimize(objective_func, vars0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13})
            
            # Accept only if strictly feasible (within numerical tolerance)
            if np.min(constraint_func(res.x)) > -1e-6:
                val = np.sum(res.x[0::3])
                if val > best_val:
                    best_val = val
                    best_sol = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local refinement around the best solution found
    if best_sol is not None:
        for refine_seed in range(15):
            np.random.seed(200 + refine_seed)
            vars0 = best_sol.copy()
            noise = np.random.normal(0, 0.002, len(vars0))
            vars0 += noise
            
            # Enforce bounds after perturbation
            for i in range(0, len(vars0), 3):
                vars0[i] = np.clip(vars0[i], 1e-5, 0.5)
                vars0[i+1] = np.clip(vars0[i+1], 0.0, 1.0)
                vars0[i+2] = np.clip(vars0[i+2], 0.0, 1.0)
                
            try:
                res = minimize(objective_func, vars0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13})
                if np.min(constraint_func(res.x)) > -1e-6:
                    val = np.sum(res.x[0::3])
                    if val > best_val:
                        best_val = val
                        best_sol = res.x.copy()
            except Exception:
                continue

    # Fallback to a valid, though suboptimal, packing if optimization fails
    if best_sol is None:
        r_f = 0.06
        centers = np.random.uniform(r_f, 1.0 - r_f, (n, 2))
        radii = np.full(n, r_f)
        return centers, radii, float(np.sum(radii))
        
    # Reconstruct centers from optimized parameters
    radii = best_sol[0::3]
    u = best_sol[1::3]
    v = best_sol[2::3]
    x = radii + u * (1.0 - 2.0 * radii)
    y = radii + v * (1.0 - 2.0 * radii)
    centers = np.column_stack((x, y))
    
    return centers, radii, float(np.sum(radii))
