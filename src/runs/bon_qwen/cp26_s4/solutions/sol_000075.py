# sol_000075 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 05693c56) state=cf24b6c5 sum of radii=2.588519 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_constraints(vars):
    """Computes inequality constraints for the packing problem."""
    n = N_CIRCLES
    c = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    cons = []
    # Boundary constraints: center +/- radius within [0, 1]
    for i in range(n):
        cons.append(c[i, 0] - r[i] - 1e-8)
        cons.append(1 - c[i, 0] - r[i] - 1e-8)
        cons.append(c[i, 1] - r[i] - 1e-8)
        cons.append(1 - c[i, 1] - r[i] - 1e-8)
        cons.append(r[i])  # Non-negativity of radii
        
    # Pairwise non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            diff = c[i] - c[j]
            dist = np.sqrt(np.sum(diff**2))
            cons.append(dist - r[i] - r[j] - 1e-8)
    return np.array(cons)

def objective(vars):
    """Objective function: negative sum of radii (for minimization)."""
    return -np.sum(vars[2*N_CIRCLES:])

def run_packing():
    """Finds optimal centers and radii for 26 circles in a unit square."""
    n = N_CIRCLES
    best_sum = -1.0
    best_result = None
    
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    constraints_def = {'type': 'ineq', 'fun': compute_constraints}
    
    # Generate multiple diverse initial configurations
    configs = []
    
    # 1. Uniform grid initialization
    c_grid = np.zeros((n, 2))
    cols = 6
    for i in range(n):
        c_grid[i] = [(i % cols + 0.5) / cols, (i // cols + 0.5) / (np.ceil(n/cols))]
    configs.append(np.concatenate([c_grid.flatten(), np.ones(n)*0.05]))
    
    # 2. Hexagonal lattice initialization
    c_hex = np.zeros((n, 2))
    idx = 0
    for r in range(6):
        for c in range(cols):
            if idx >= n: break
            x = (c + 0.5 + 0.5*(r%2)) / cols
            y = (r + 0.5) / 5.0
            c_hex[idx] = [x, y]
            idx += 1
    configs.append(np.concatenate([c_hex.flatten(), np.ones(n)*0.05]))
    
    # 3. Random initializations (3 restarts)
    for seed in [42, 123, 999]:
        np.random.seed(seed)
        c_rand = np.random.rand(n, 2) * 0.8 + 0.1
        configs.append(np.concatenate([c_rand.flatten(), np.ones(n)*0.04]))
        
    # Run optimization for each configuration
    for x0 in configs:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints_def, options={'maxiter': 1500, 'ftol': 1e-10})
            
            if -res.fun > best_sum:
                centers_opt = res.x[:2*n].reshape(n, 2)
                radii_opt = res.x[2*n:]
                
                # Validate against strict problem constraints
                valid = True
                for i in range(n):
                    if (centers_opt[i,0] < radii_opt[i] - 1e-6 or 
                        centers_opt[i,0] + radii_opt[i] > 1 + 1e-6 or
                        centers_opt[i,1] < radii_opt[i] - 1e-6 or 
                        centers_opt[i,1] + radii_opt[i] > 1 + 1e-6):
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i+1, n):
                            d = np.linalg.norm(centers_opt[i] - centers_opt[j])
                            if d < radii_opt[i] + radii_opt[j] - 1e-6:
                                valid = False
                                break
                        if not valid: break
                    if valid:
                        best_sum = -res.fun
                        best_result = (centers_opt.copy(), radii_opt.copy())
        except Exception:
            continue
            
    # Fallback in case optimization fails to find a valid configuration
    if best_result is None:
        centers_fb = np.random.rand(n, 2) * 0.6 + 0.2
        radii_fb = np.ones(n) * 0.05
        return centers_fb, radii_fb, np.sum(radii_fb)
        
    return best_result[0], best_result[1], best_sum
