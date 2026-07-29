# sol_000288 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a4dfceb8) state=2cdc1f1e sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def obj_func(vars):
    """Objective function to maximize radius r (minimize -r)."""
    return -vars[-1]

def constraint_func(vars):
    """
    Computes inequality constraints:
    - Boundary: x >= r, 1-x >= r, y >= r, 1-y >= r
    - Overlap: dist(i,j) >= 2r
    Returns an array of constraint values (must be >= 0).
    """
    n = 26
    c = vars[:-2].reshape((n, 2))
    r = vars[-1]
    
    # Boundary constraints (4 per circle)
    b_cons = np.concatenate([
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r
    ])
    
    # Overlap constraints (vectorized pairwise distances)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu, iv = np.triu_indices(n, k=1)
    o_cons = dists[iu, iv] - 2.0*r
    
    return np.concatenate([b_cons, o_cons])

def generate_initial(n):
    """Generates a staggered grid initial layout approximating hexagonal packing."""
    centers = np.zeros((n, 2))
    idx = 0
    cols = 5
    rows = 6
    for r_idx in range(rows):
        for c_idx in range(cols):
            if idx < n:
                x = (c_idx + 0.5) / cols
                y = (r_idx + 0.5) / rows
                if r_idx % 2 == 1:
                    x += 1.0 / (2 * cols)
                centers[idx] = [x, y]
                idx += 1
    return centers

def run_packing():
    np.random.seed(42)
    n = 26
    best_r = 0.0
    best_centers = None
    
    # Variable bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)]
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Multiple restarts to find global optimum
    for trial in range(5):
        centers = generate_initial(n)
        # Add random perturbation to escape grid symmetry
        centers += np.random.uniform(-0.04, 0.04, size=centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        # Initialize with a feasible small radius
        vars0 = np.concatenate([centers.flatten(), [0.08]])
        
        try:
            res = minimize(obj_func, vars0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
            if -res.fun > best_r:
                best_r = -res.fun
                best_centers = res.x[:-2].reshape((n, 2))
        except Exception:
            continue
            
    # Fallback if optimization fails
    if best_centers is None:
        best_centers = generate_initial(n)
        best_r = 0.08
        
    # Post-processing to guarantee strict validity within tolerance
    c = best_centers
    r = best_r
    min_d = 2.0
    
    # Check boundary distances
    for i in range(n):
        x, y = c[i]
        min_d = min(min_d, x, 1.0-x, y, 1.0-y)
        
    # Check pairwise distances
    for i in range(n):
        for j in range(i+1, n):
            dist = np.sqrt(np.sum((c[i]-c[j])**2))
            min_d = min(min_d, dist)
            
    # Shrink slightly to account for numerical tolerance in validator
    r_final = min(r, min_d / 2.0) * 0.999999
    radii = np.full(n, r_final)
    
    return best_centers, radii, r_final * n
