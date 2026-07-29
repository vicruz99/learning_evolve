# sol_000087 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 47f2e0af) state=9a91dc19 sum of radii=2.556856 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Objective function: minimize negative sum of radii (equivalent to maximizing sum)."""
    return -np.sum(vars[2::3])

def constraints_func(vars, n):
    """Compute all inequality constraints: boundary and non-overlap."""
    c = []
    # Boundary constraints
    for i in range(n):
        xi = vars[3*i]
        yi = vars[3*i+1]
        ri = vars[3*i+2]
        c.append(xi - ri)
        c.append(1.0 - xi - ri)
        c.append(yi - ri)
        c.append(1.0 - yi - ri)
        
    # Pairwise non-overlap constraints
    for i in range(n):
        xi, yi = vars[3*i], vars[3*i+1]
        ri = vars[3*i+2]
        for j in range(i+1, n):
            xj, yj = vars[3*j], vars[3*j+1]
            rj = vars[3*j+2]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            c.append(dist - ri - rj)
            
    return np.array(c)

def constraints_wrapper(vars):
    """Wrapper to pass fixed N to constraints_func without closures."""
    return constraints_func(vars, N)

def generate_init(n):
    """Generate initial guess using a hexagonal-like arrangement."""
    centers = np.zeros((n, 2))
    idx = 0
    # Row configuration sums to 26 circles
    rows = [5, 4, 5, 4, 5, 3]
    y = 0.12
    for i, count in enumerate(rows):
        # Shift odd rows to create hexagonal packing
        x_start = 0.12 if i % 2 == 0 else 0.22
        for j in range(count):
            centers[idx, 0] = x_start + j * 0.19
            centers[idx, 1] = y
            idx += 1
        y += 0.155
        
    # Initial radius: small but reasonable to allow growth
    r = 0.05 + 0.02 * np.random.random()
    vars = np.zeros(3*n)
    for i in range(n):
        vars[3*i] = centers[i,0]
        vars[3*i+1] = centers[i,1]
        vars[3*i+2] = r
    return vars

def run_packing():
    """Optimize circle packing to maximize sum of radii."""
    np.random.seed(42)
    n = N
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints_wrapper}
    
    best_sum = -1.0
    best_x = None
    
    # Multiple restarts to avoid local minima
    for _ in range(8):
        x0 = generate_init(n)
        # Add small random perturbation to break symmetry
        x0 = x0 + np.random.normal(0, 0.005, size=3*n)
        # Ensure initial guess respects bounds
        x0[:2*n] = np.clip(x0[:2*n], 0.0, 1.0)
        x0[2*n:] = np.clip(x0[2*n:], 0.0, 0.5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            
            if res.success:
                current_sum = -res.fun
                # Verify constraints are satisfied within numerical tolerance
                c_val = constraints_func(res.x, n)
                if np.min(c_val) >= -1e-7:
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_x = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization fails completely
    if best_x is None:
        x0 = generate_init(n)
        best_x = x0
        best_sum = -objective(x0)
        
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        radii[i] = best_x[3*i+2]
        
    return centers, radii, np.sum(radii)
