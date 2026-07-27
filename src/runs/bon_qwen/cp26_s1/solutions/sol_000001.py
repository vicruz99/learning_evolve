# sol_000001 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6882cd8b) state=69171a42 sum of radii=2.483556 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    np.random.seed(42)
    n = 26
    best_r = 0.0
    best_c = None
    
    # Try multiple initial configurations to escape local optima
    for seed in range(15):
        np.random.seed(seed)
        
        # Generate points on a hexagonal lattice (optimal packing geometry)
        pts = []
        for i in range(8):
            for j in range(6):
                x = i + 0.5 * (j % 2)
                y = j * np.sqrt(3) / 2.0
                pts.append([x, y])
        
        # Randomly select n points to form initial centers
        idx = np.random.choice(len(pts), size=n, replace=False)
        centers = np.array([pts[i] for i in idx])
        
        # Normalize and center in [0, 1]^2 with a margin
        mn = centers.min(axis=0)
        mx = centers.max(axis=0)
        span = mx - mn
        if np.any(span == 0):
            continue
        centers = (centers - mn) / span
        centers *= 0.85 
        centers += 0.075
        
        r_init = 0.08
        
        def objective(vars):
            # Maximize r => minimize -r
            return -vars[52]
            
        def constraints(vars):
            c = vars[:52].reshape(n, 2)
            r = vars[52]
            
            # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
            b = np.concatenate([
                c[:, 0] - r,
                1.0 - c[:, 0] - r,
                c[:, 1] - r,
                1.0 - c[:, 1] - r
            ])
            
            # Pairwise non-overlap: dist^2 >= 4r^2
            # Vectorized computation for efficiency
            i_idx, j_idx = np.triu_indices(n, k=1)
            dx = c[i_idx, 0] - c[j_idx, 0]
            dy = c[i_idx, 1] - c[j_idx, 1]
            p = dx*dx + dy*dy - 4.0*r*r
            
            return np.concatenate([b, p])
            
        cons = {'type': 'ineq', 'fun': constraints}
        bounds = [(0.0, 1.0)] * 52 + [(0.0, 0.5)]
        
        x0 = np.zeros(53)
        x0[:52] = centers.flatten()
        x0[52] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', constraints=cons, bounds=bounds,
                           options={'maxiter': 800, 'ftol': 1e-9, 'disp': False})
            
            if res.success or (-res.fun > best_r):
                # Verify constraints are met within numerical tolerance
                c_vals = constraints(res.x)
                if np.min(c_vals) > -1e-6:
                    r_opt = -res.fun
                    if r_opt > best_r:
                        best_r = r_opt
                        best_c = res.x[:52].reshape(n, 2)
        except Exception:
            continue
            
    if best_c is None:
        # Fallback to a structured grid if optimization fails
        best_c = np.column_stack(np.mgrid[0:6, 0:5].reshape(2, -1))[:26] / 6.0 * 0.8 + 0.1
        best_r = 0.05
        
    centers = best_c
    radii = np.full(n, best_r)
    return centers, radii, np.sum(radii)
