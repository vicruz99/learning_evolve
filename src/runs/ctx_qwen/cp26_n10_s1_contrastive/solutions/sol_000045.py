# sol_000045 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000003 (state f9d5c394) state=ee6daed9 sum of radii=2.609515 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def obj_func(x):
    """Objective: minimize negative sum of radii."""
    return -np.sum(x[0::3])

def constr_func(x):
    """
    Constraint function: pairwise non-overlap.
    Returns array of values that must be >= 0.
    Uses transformed coordinates to automatically satisfy boundary constraints.
    """
    r = x[0::3]
    u = x[1::3]
    v = x[2::3]
    
    # Transform normalized coordinates to actual positions
    cx = r + u * (1.0 - 2.0 * r)
    cy = r + v * (1.0 - 2.0 * r)
    
    # Compute squared distances between all pairs
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist_sq = dx**2 + dy**2
    
    # Compute squared minimum allowed distances
    r_sum = r[:, None] + r[None, :]
    min_dist_sq = r_sum**2
    
    # Extract upper triangular part (i < j) to avoid duplicates and self-comparison
    i, j = np.triu_indices(N, k=1)
    return dist_sq[i, j] - min_dist_sq[i, j]

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Variable layout: [r0, u0, v0, r1, u1, v1, ..., r25, u25, v25]
    bounds = [(1e-5, 0.5), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constr_func}
    
    best_val = -np.inf
    best_vars = None
    
    inits = []
    lower = [1e-5]*N + [0.0]*N + [0.0]*N
    upper = [0.5]*N + [1.0]*N + [1.0]*N
    
    # 1. Random initializations to explore space
    for seed in range(20):
        np.random.seed(seed)
        r0 = 0.06 + np.random.rand() * 0.04
        u0 = np.random.rand(N)
        v0 = np.random.rand(N)
        vars0 = np.empty(N * 3)
        vars0[0::3] = r0
        vars0[1::3] = u0
        vars0[2::3] = v0
        inits.append(vars0)
        
    # 2. Hexagonal lattice initializations (theoretically dense)
    for r0 in [0.075, 0.085, 0.095, 0.105]:
        centers = []
        y = r0
        row = 0
        while len(centers) < N:
            x = r0 if row % 2 == 0 else r0 + r0 * 0.5
            while x <= 1.0 - r0 and len(centers) < N:
                centers.append((x, y))
                x += 2.0 * r0
            y += np.sqrt(3.0) * r0
            row += 1
            
        centers = np.array(centers[:N])
        denom = 1.0 - 2.0 * r0
        u0 = (centers[:, 0] - r0) / denom
        v0 = (centers[:, 1] - r0) / denom
        u0 = np.clip(u0, 0.05, 0.95)
        v0 = np.clip(v0, 0.05, 0.95)
        
        vars0 = np.empty(N * 3)
        vars0[0::3] = r0
        vars0[1::3] = u0
        vars0[2::3] = v0
        inits.append(vars0)
        
    # Global search phase
    for idx, vars0 in enumerate(inits):
        # Perturb to break symmetry
        np.random.seed(idx + 100)
        vars0 += np.random.normal(0, 0.008, len(vars0))
        vars0 = np.clip(vars0, lower, upper)
        
        try:
            res = minimize(obj_func, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13})
            
            if res.success:
                c_min = np.min(constr_func(res.x))
                if c_min >= -1e-7:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # Local refinement phase on the best configuration found
    if best_vars is not None:
        for rep in range(5):
            np.random.seed(rep + 200)
            vars_ref = best_vars + np.random.normal(0, 0.005, len(best_vars))
            vars_ref = np.clip(vars_ref, lower, upper)
            
            try:
                res = minimize(obj_func, vars_ref, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-14})
                if res.success and np.min(constr_func(res.x)) >= -1e-7:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_vars = res.x.copy()
            except Exception:
                continue
                
    # Fallback if optimization unexpectedly fails
    if best_vars is None:
        r_f = 0.05
        indices = np.arange(N)
        cols = indices % 6
        rows = indices // 6
        x_f = 0.1 + cols * 0.15
        y_f = 0.1 + rows * 0.15
        centers = np.column_stack((x_f, y_f))
        radii = np.full(N, r_f)
        return centers, radii, np.sum(radii)
        
    # Reconstruct final centers and radii from optimized parameters
    radii = best_vars[0::3]
    u = best_vars[1::3]
    v = best_vars[2::3]
    x = radii + u * (1.0 - 2.0 * radii)
    y = radii + v * (1.0 - 2.0 * radii)
    centers = np.column_stack((x, y))
    
    return centers, radii, float(np.sum(radii))
