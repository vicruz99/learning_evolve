# sol_000220 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96713eb2) state=19044109 sum of radii=2.484698 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(vars):
    """Objective: maximize common radius r (minimize -r)"""
    return -vars[-1]

def _con_boundary(vars):
    """Constraint: circles must remain inside [0,1]x[0,1]"""
    n = (len(vars) - 1) // 2
    r = vars[-1]
    x = vars[::2]
    y = vars[1::2]
    # x >= r, x <= 1-r, y >= r, y <= 1-r  =>  >= 0
    return np.concatenate([x - r, 1 - r - x, y - r, 1 - r - y])

def _con_overlap(vars):
    """Constraint: no overlapping circles (dist^2 >= 4r^2)"""
    n = (len(vars) - 1) // 2
    r = vars[-1]
    x = vars[::2]
    y = vars[1::2]
    i, j = np.triu_indices(n, k=1)
    dx = x[i] - x[j]
    dy = y[i] - y[j]
    return dx**2 + dy**2 - 4*r**2

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Multi-start strategy for robustness against local minima
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        centers = np.zeros((n, 2))
        idx = 0
        # Hexagonal row distribution for 26 circles
        rows = [5, 6, 5, 6, 4]
        r_init = 0.09
        for i, cnt in enumerate(rows):
            y = r_init + i * r_init * np.sqrt(3)
            x_start = r_init if i % 2 == 0 else r_init + r_init
            for j in range(cnt):
                x = x_start + j * 2 * r_init
                if idx < n:
                    centers[idx] = [x, y]
                    idx += 1
        while idx < n:
            centers[idx] = [0.5, 0.5]
            idx += 1
            
        # Add small random perturbation to break symmetry
        centers += np.random.randn(n, 2) * 0.003
        centers = np.clip(centers, 0.02, 0.98)
        
        bounds = [(0, 1)] * (2*n) + [(0, 0.5)]
        cons = [{'type': 'ineq', 'fun': _con_boundary},
                {'type': 'ineq', 'fun': _con_overlap}]
                
        x0 = np.concatenate([centers.flatten(), [r_init]])
        
        try:
            res = minimize(_objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
            r_opt = res.x[-1]
            c_opt = res.x[:-1].reshape(n, 2)
            
            # Post-processing: ensure strict validity within tolerance
            for _ in range(50):
                valid = True
                for i in range(n):
                    if c_opt[i,0] - r_opt < -1e-9 or c_opt[i,0] + r_opt > 1 + 1e-9:
                        r_opt *= 0.999; valid = False; break
                    if c_opt[i,1] - r_opt < -1e-9 or c_opt[i,1] + r_opt > 1 + 1e-9:
                        r_opt *= 0.999; valid = False; break
                if not valid: continue
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.sqrt(np.sum((c_opt[i] - c_opt[j])**2))
                        if d < 2*r_opt - 1e-9:
                            r_opt *= 0.999; valid = False; break
                    if not valid: break
                if valid: break
                
            current_sum = 26 * r_opt
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = c_opt.copy()
                best_radii = np.full(n, r_opt)
        except Exception:
            continue
            
    if best_centers is None:
        # Fallback safety net (should not be reached with proper optimization)
        return np.zeros((n, 2)), np.zeros(n), 0.0
        
    return best_centers, best_radii, best_sum
