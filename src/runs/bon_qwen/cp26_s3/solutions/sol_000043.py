# sol_000043 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9e7c8308) state=46a2cdbb sum of radii=2.617879 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars):
    n = len(vars) // 3
    return -np.sum(vars[2*n:])

def constr_ineq(vars):
    n = len(vars) // 3
    xs = vars[:n]
    ys = vars[n:2*n]
    rs = vars[2*n:]
    
    # Preallocate constraint array: 4 boundary constraints per circle + n*(n-1)/2 overlap constraints
    n_constraints = 4*n + n*(n-1)//2
    cons = np.empty(n_constraints)
    idx = 0
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    cons[idx:idx+n] = xs - rs; idx += n
    cons[idx:idx+n] = 1.0 - xs - rs; idx += n
    cons[idx:idx+n] = ys - rs; idx += n
    cons[idx:idx+n] = 1.0 - ys - rs; idx += n
    
    # Overlap constraints: distance >= r_i + r_j
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            cons[idx] = d - (rs[i] + rs[j])
            idx += 1
            
    return cons

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None

    # Bounds: x,y in [0,1], r in [1e-6, 0.5]
    bounds = [(0, 1)]*n + [(0, 1)]*n + [(1e-6, 0.5)]*n
    cons_dict = {'type': 'ineq', 'fun': constr_ineq}
    
    # Multiple restarts to find global optimum
    for seed in range(20):
        np.random.seed(seed)
        # Random initialization in the inner square to avoid boundary issues initially
        cx = np.random.rand(n) * 0.8 + 0.1
        cy = np.random.rand(n) * 0.8 + 0.1
        r = 0.08 * np.ones(n)
        x0 = np.concatenate([cx, cy, r])
        
        try:
            res = minimize(objective, x0, method='SLSQP', 
                           bounds=bounds, constraints=cons_dict, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            
            cx_opt = res.x[:n]
            cy_opt = res.x[n:2*n]
            r_opt = res.x[2*n:]
            
            # Strict validation matching the problem's tolerance
            valid = True
            if np.any(cx_opt < r_opt - 1e-9) or np.any(cx_opt > 1 - r_opt + 1e-9):
                valid = False
            if np.any(cy_opt < r_opt - 1e-9) or np.any(cy_opt > 1 - r_opt + 1e-9):
                valid = False
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.sqrt((cx_opt[i]-cx_opt[j])**2 + (cy_opt[i]-cy_opt[j])**2)
                        if d < r_opt[i] + r_opt[j] - 1e-9:
                            valid = False
                            break
                    if not valid: break
            
            if valid:
                s = np.sum(r_opt)
                if s > best_sum:
                    best_sum = s
                    best_centers = np.column_stack([cx_opt, cy_opt])
                    best_radii = r_opt
        except Exception:
            continue

    # Fallback solution if optimization fails
    if best_centers is None:
        cx = np.tile(np.linspace(0.1, 0.9, 5), 5)
        cy = np.repeat(np.linspace(0.1, 0.9, 5), 5)
        cx = cx[:n]
        cy = cy[:n]
        r = 0.05 * np.ones(n)
        best_centers = np.column_stack([cx, cy])
        best_radii = r
        best_sum = np.sum(r)
        
    return best_centers, best_radii, best_sum
