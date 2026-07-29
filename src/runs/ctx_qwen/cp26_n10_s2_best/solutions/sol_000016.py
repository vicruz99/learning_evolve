# sol_000016 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7b3d1146) state=585439f0 sum of radii=2.595191 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_objective(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2*N_CIRCLES:])

def compute_constraints(vars):
    """Inequality constraints: boundaries and non-overlap."""
    n = N_CIRCLES
    x = vars[:n]
    y = vars[n:2*n]
    r = vars[2*n:]
    
    # Total constraints: 4 boundary per circle + 1 per pair
    m = 4 * n + n * (n - 1) // 2
    c = np.empty(m)
    idx = 0
    
    # Boundary constraints
    for i in range(n):
        c[idx] = x[i] - r[i]; idx += 1
        c[idx] = 1.0 - x[i] - r[i]; idx += 1
        c[idx] = y[i] - r[i]; idx += 1
        c[idx] = 1.0 - y[i] - r[i]; idx += 1
        
    # Non-overlap constraints
    for i in range(n):
        xi, yi, ri = x[i], y[i], r[i]
        for j in range(i + 1, n):
            dist = np.sqrt((xi - x[j])**2 + (yi - y[j])**2)
            c[idx] = dist - ri - r[j]
            idx += 1
            
    return c

def run_packing():
    n = N_CIRCLES
    best_val = -np.inf
    best_x = None
    
    # Variable bounds: coordinates in [0, 1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    
    np.random.seed(42)
    
    # Multi-start optimization from hexagonal lattice perturbations
    for trial in range(8):
        rows = [6, 5, 6, 5, 4]  # Sums to 26
        x_init, y_init, r_init = [], [], []
        idx = 0
        for r_idx, count in enumerate(rows):
            y_val = 0.08 + r_idx * 0.17
            x_start = 0.08 if r_idx % 2 == 0 else 0.17
            for c_idx in range(count):
                if idx >= n: break
                x_val = x_start + c_idx * 0.17
                # Add controlled noise to break symmetry and explore space
                x_init.append(np.clip(x_val + np.random.uniform(-0.03, 0.03), 0.05, 0.95))
                y_init.append(np.clip(y_val + np.random.uniform(-0.03, 0.03), 0.05, 0.95))
                r_init.append(0.05)
                idx += 1
                
        x0 = np.array(x_init + y_init + r_init)
        
        res = minimize(compute_objective, x0, method='SLSQP',
                       constraints={'type': 'ineq', 'fun': compute_constraints},
                       bounds=bounds,
                       options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
                       
        # Accept if better and sufficiently feasible
        if -res.fun > best_val:
            if np.all(compute_constraints(res.x) >= -1e-8):
                best_val = -res.fun
                best_x = res.x.copy()
                
    centers = np.column_stack((best_x[:n], best_x[n:2*n]))
    radii = best_x[2*n:]
    return centers, radii, float(np.sum(radii))
