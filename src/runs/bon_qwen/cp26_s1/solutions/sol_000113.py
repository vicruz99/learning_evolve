# sol_000113 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a98c42c6) state=08ca9cdd sum of radii=2.568273 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_constraints(centers, radii):
    """Compute constraint margins for boundaries and non-overlap."""
    n = len(radii)
    num_constraints = 4 * n + n * (n - 1) // 2
    con = np.empty(num_constraints)
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    con[:n] = centers[:, 0] - radii
    con[n:2*n] = 1.0 - centers[:, 0] - radii
    con[2*n:3*n] = centers[:, 1] - radii
    con[3*n:4*n] = 1.0 - centers[:, 1] - radii
    
    # Non-overlap constraints: squared distance >= squared sum of radii
    diff = centers[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff**2, axis=2)
    r_sum = radii[:, None] + radii[None, :]
    
    iu, ju = np.triu_indices(n, k=1)
    con[4*n:] = d2[iu, ju] - r_sum[iu, ju]**2
    return con

def objective(vars):
    """Objective function: negative sum of radii (for minimization)."""
    return -np.sum(vars[52:])

def constraint(vars):
    """Constraint function wrapper for SLSQP."""
    centers = vars[:52].reshape(26, 2)
    radii = vars[52:]
    return get_constraints(centers, radii)

def run_packing():
    n = 26
    best_sum = -1.0
    best_vars = None
    
    # Prepare multiple initializations to avoid poor local minima
    inits = []
    
    # Init 1: Grid-based layout with an extra point
    xs = np.linspace(0.15, 0.85, 5)
    ys = np.linspace(0.15, 0.85, 5)
    c1 = np.array([[x, y] for y in ys for x in xs])
    c1 = np.vstack([c1, [0.5, 0.5]])
    r1 = np.full(n, 0.04)
    inits.append((c1, r1))
    
    # Init 2: Randomized layout (Seed 42)
    np.random.seed(42)
    c2 = np.random.uniform(0.12, 0.88, (n, 2))
    r2 = np.full(n, 0.035)
    inits.append((c2, r2))
    
    # Init 3: Randomized layout (Seed 123)
    np.random.seed(123)
    c3 = np.random.uniform(0.12, 0.88, (n, 2))
    r3 = np.full(n, 0.035)
    inits.append((c3, r3))
    
    # Bounds for variables: centers in [0,1], radii in [1e-4, 0.5]
    bounds = [(0.0, 1.0)] * 52 + [(1e-4, 0.5)] * 26
    cons = {'type': 'ineq', 'fun': constraint}
    
    # Run optimization for each initialization
    for c0, r0 in inits:
        vars0 = np.concatenate([c0.flatten(), r0])
        try:
            res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
            current_sum = -res.fun
            if res.success and current_sum > best_sum:
                best_sum = current_sum
                best_vars = res.x.copy()
        except Exception:
            pass
            
    # Fallback if optimization fails
    if best_vars is None:
        centers = np.random.uniform(0.2, 0.8, (n, 2))
        radii = np.full(n, 0.02)
        return centers, radii, np.sum(radii)
        
    centers = best_vars[:52].reshape(n, 2)
    radii = best_vars[52:]
    
    # Final safety clamp to guarantee strict validity within numerical tolerance
    # SLSQP should satisfy constraints, but we enforce a tiny margin for robustness
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], radii[i], 1.0 - radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], 1.0 - radii[i])
        
    return centers, radii, best_sum
