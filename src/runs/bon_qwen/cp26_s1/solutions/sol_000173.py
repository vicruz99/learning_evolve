# sol_000173 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8150d860) state=5b4c9734 sum of radii=0.001300 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Structured grid initialization (5x5 + 1)
    gs = np.linspace(0.15, 0.85, 5)
    c_struct = np.array([[x, y] for x in gs for y in gs])
    c_struct = np.vstack([c_struct, [0.9, 0.1]])
    inits.append(c_struct)
    
    # 2-6. Random initializations
    for _ in range(5):
        inits.append(rng.uniform(0.1, 0.9, (n, 2)))
        
    for centers_init in inits:
        radii_init = np.full(n, 0.08)
        v0 = np.concatenate([centers_init.flatten(), radii_init])
        bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
        
        def objective(v, lam):
            x = v[:n]
            y = v[n:2*n]
            r = v[2*n:]
            
            # Primary objective: maximize sum of radii
            base = -np.sum(r)
            
            # Boundary penalties: circles must stay within [0,1]^2
            v1 = np.maximum(0, r - x)
            v2 = np.maximum(0, x + r - 1)
            v3 = np.maximum(0, r - y)
            v4 = np.maximum(0, y + r - 1)
            b_pen = np.sum(v1**2 + v2**2 + v3**2 + v4**2)
            
            # Overlap penalties: distance between centers >= sum of radii
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2)
            overlap = (r[:, None] + r[None, :]) - dist
            # Sum over upper triangle only (divide full sum by 2)
            p_pen = np.sum(np.maximum(0, overlap)**2) / 2.0
            
            return base + lam * (b_pen + p_pen)
            
        # Phase 1: Moderate penalty to find dense configuration
        res1 = minimize(lambda v: objective(v, 500.0), v0, method='L-BFGS-B', 
                        bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12})
                        
        # Phase 2: High penalty to strictly enforce constraints
        res2 = minimize(lambda v: objective(v, 5000.0), res1.x, method='L-BFGS-B', 
                        bounds=bounds, options={'maxiter': 3000, 'ftol': 1e-12})
        
        v_opt = res2.x
        r_opt = v_opt[2*n:]
        curr_sum = np.sum(r_opt)
        
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = np.column_stack([v_opt[:n], v_opt[n:2*n]])
            best_radii = r_opt
            
    return best_centers, best_radii, best_sum
