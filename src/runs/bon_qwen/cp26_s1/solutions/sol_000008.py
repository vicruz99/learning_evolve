# sol_000008 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e6663bde) state=fef51320 sum of radii=2.620431 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints(vars):
    n = 26
    radii = vars[2*n:]
    nc = 4*n + n*(n-1)//2
    c = np.empty(nc)
    idx = 0
    
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c[idx:idx+n] = vars[:n] - radii
    idx += n
    c[idx:idx+n] = 1.0 - vars[:n] - radii
    idx += n
    c[idx:idx+n] = vars[n:2*n] - radii
    idx += n
    c[idx:idx+n] = 1.0 - vars[n:2*n] - radii
    idx += n
    
    # Non-overlap constraints: dist^2 >= (r1 + r2)^2
    for i in range(n):
        xi, yi = vars[i], vars[n+i]
        ri = radii[i]
        for j in range(i+1, n):
            dx = xi - vars[j]
            dy = yi - vars[n+j]
            c[idx] = dx*dx + dy*dy - (ri + radii[j])**2
            idx += 1
    return c

def constraint_func(vars):
    return compute_constraints(vars)

def objective(vars):
    n = 26
    return -np.sum(vars[2*n:])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_centers = None
    best_radii = None
    max_sum_r = -1.0
    
    bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    for seed in range(40):
        np.random.seed(seed)
        
        if seed < 15:
            # Random initialization in central region
            x = np.random.uniform(0.2, 0.8, n)
            y = np.random.uniform(0.2, 0.8, n)
        elif seed < 25:
            # Hexagonal-inspired initialization
            x, y = [], []
            row_counts = [5, 6, 5, 6, 4]
            for r_idx, cnt in enumerate(row_counts):
                if len(x) + cnt > n: cnt = n - len(x)
                y_val = 0.2 + r_idx * 0.16
                offset = 0.08 if r_idx % 2 == 1 else 0.0
                x_vals = np.linspace(0.15 + offset, 0.85 + offset, cnt)
                x.extend(x_vals)
                y.extend([y_val]*cnt)
            x, y = np.array(x[:n]), np.array(y[:n])
        else:
            # Wider random initialization
            x = np.random.uniform(0.1, 0.9, n)
            y = np.random.uniform(0.1, 0.9, n)
            
        r_init = np.full(n, 0.06)
        vars0 = np.concatenate([x, y, r_init])
        
        try:
            res = minimize(objective, vars0, method='SLSQP', 
                          bounds=bounds, constraints=cons, 
                          options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
            
            curr_sum = -res.fun
            if curr_sum > max_sum_r:
                c_vals = compute_constraints(res.x)
                if np.all(c_vals >= -1e-6):
                    max_sum_r = curr_sum
                    best_centers = np.column_stack((res.x[:n], res.x[n:2*n])).copy()
                    best_radii = res.x[2*n:].copy()
        except Exception:
            continue
            
    if best_centers is None:
        # Fallback to structured grid
        best_centers = np.zeros((n, 2))
        best_radii = np.full(n, 0.05)
        idx = 0
        for i in range(5):
            for j in range(6):
                if idx < n:
                    best_centers[idx] = [0.1 + j*0.15, 0.1 + i*0.2]
                    idx += 1
        max_sum_r = np.sum(best_radii)
        
    return best_centers, best_radii, max_sum_r
