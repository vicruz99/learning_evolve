# sol_000114 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 028484b6) state=881f8e0e sum of radii=1.040000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(x, n):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    return -np.sum(x[2*n:])

def _constraints_func(x, n):
    """Constraint function: returns array of constraint values >= 0."""
    c = x[:2*n].reshape(n, 2)
    r = x[2*n:]
    vals = []
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    for i in range(n):
        vals.append(c[i, 0] - r[i])
        vals.append(1.0 - c[i, 0] - r[i])
        vals.append(c[i, 1] - r[i])
        vals.append(1.0 - c[i, 1] - r[i])
        
    # Pairwise non-overlap constraints: distance >= sum of radii
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = np.sum((c[i] - c[j])**2)
            vals.append(dist_sq - (r[i] + r[j])**2)
            
    return np.array(vals)

def run_packing():
    n = 26
    best_sum = 0.0
    best_c = None
    best_r = None
    
    # Configuration parameters
    n_trials = 5
    max_iter = 3000
    
    for trial in range(n_trials):
        np.random.seed(42 + trial)
        centers = np.zeros((n, 2))
        
        # Hexagonal-like initial layout
        row_counts = [5, 6, 5, 6, 4]
        idx = 0
        for r_idx, count in enumerate(row_counts):
            y = 0.12 + r_idx * 0.19
            for c_idx in range(count):
                x = 0.06 + c_idx * 0.175
                if r_idx % 2 == 1:
                    x += 0.0875
                centers[idx] = [x, y]
                idx += 1
                
        # Add random perturbation and clip to valid interior
        centers += np.random.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        
        # Initial small radii to ensure constraints are satisfied
        radii = np.full(n, 0.04)
        
        # Flatten variables: [x1, y1, ..., x26, y26, r1, ..., r26]
        x0 = np.concatenate([centers.flatten(), radii])
        
        # Bounds: coordinates in [0,1], radii in [0, 0.5]
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
        
        # Constraint dictionary
        cons = {
            'type': 'ineq',
            'fun': _constraints_func,
            'args': (n,)
        }
        
        try:
            res = minimize(
                _objective, x0, args=(n,), method='SLSQP',
                bounds=bounds, constraints=cons,
                options={'maxiter': max_iter, 'ftol': 1e-10, 'disp': False}
            )
            
            if res.success:
                c_opt = res.x[:2*n].reshape(n, 2)
                r_opt = res.x[2*n:]
                
                # Validate against the provided function
                if validate_packing(c_opt, r_opt):
                    curr_sum = np.sum(r_opt)
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_c = c_opt.copy()
                        best_r = r_opt.copy()
        except Exception:
            continue
            
    # Fallback to initial configuration if optimization failed entirely
    if best_c is None:
        best_c = centers
        best_r = radii
        best_sum = np.sum(best_r)
        
    return best_c, best_r, best_sum
