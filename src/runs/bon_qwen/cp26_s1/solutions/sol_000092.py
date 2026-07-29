# sol_000092 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1e7a6456) state=15827374 sum of radii=2.611574 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(v, n):
    """Objective function: minimize negative sum of radii"""
    return -np.sum(v[2::3])

def compute_constraints(v, n):
    """Compute inequality constraints: boundary and non-overlap"""
    c = []
    # Boundary constraints for each circle
    for i in range(n):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        c.append(x - r)          # x >= r
        c.append(1 - x - r)      # x + r <= 1
        c.append(y - r)          # y >= r
        c.append(1 - y - r)      # y + r <= 1
        
    # Pairwise non-overlap constraints
    for i in range(n):
        for j in range(i+1, n):
            dx = v[3*i] - v[3*j]
            dy = v[3*i+1] - v[3*j+1]
            dr = v[3*i+2] + v[3*j+2]
            c.append(dx*dx + dy*dy - dr*dr)  # dist^2 >= (r1+r2)^2
            
    return np.array(c)

def constraint_func(v):
    """Wrapper for constraints function to match scipy signature"""
    return compute_constraints(v, 26)

def run_packing():
    n = 26
    best_v = None
    best_score = float('-inf')
    
    # Multiple random restarts to escape local optima
    for seed in range(15):
        np.random.seed(seed)
        # Initial guess: random centers, small radii to ensure feasibility
        v0 = np.random.rand(3 * n)
        v0[2::3] = 0.02 * np.ones(n)  # radii
        v0[0::3] = 0.1 + 0.8 * v0[0::3]  # x centers
        v0[1::3] = 0.1 + 0.8 * v0[1::3]  # y centers
        
        bounds = [(0, 1), (0, 1), (0, 0.5)] * n
        cons = {'type': 'ineq', 'fun': constraint_func}
        
        try:
            res = minimize(compute_objective, v0, args=(n,), method='SLSQP', 
                          bounds=bounds, constraints=cons, 
                          options={'maxiter': 400, 'ftol': 1e-10})
            
            if res.success:
                score = -res.fun
                if score > best_score:
                    best_score = score
                    best_v = res.x
        except Exception:
            continue
            
    # Fallback initialization if optimization fails
    if best_v is None:
        best_v = np.zeros(3 * n)
        for i in range(n):
            best_v[3*i] = 0.25 + 0.5 * (i % 5)
            best_v[3*i+1] = 0.25 + 0.5 * (i // 5)
            best_v[3*i+2] = 0.01
            
    centers = best_v.reshape(n, 3)[:, :2]
    radii = best_v.reshape(n, 3)[:, 2]
    
    # Final numerical safety clamp
    centers = np.clip(centers, 0.0, 1.0)
    radii = np.clip(radii, 0.0, 0.5)
    
    return centers, radii, best_score
