# sol_000059 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 608ae89b) state=31ee7bab sum of radii=2.598500 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(params):
    """Minimize negative sum of radii to maximize total radius."""
    return -np.sum(params[2 * N_CIRCLES:])

def get_constraints(params):
    """Compute all inequality constraints for SLSQP (must be >= 0)."""
    c = params[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    r = params[2 * N_CIRCLES:]
    cons = []
    
    # Boundary constraints: 0 <= c - r  and  c + r <= 1
    for i in range(N_CIRCLES):
        cons.append(c[i, 0] - r[i])
        cons.append(c[i, 1] - r[i])
        cons.append(1.0 - c[i, 0] - r[i])
        cons.append(1.0 - c[i, 1] - r[i])
        
    # Non-overlap constraints: ||c_i - c_j||^2 >= (r_i + r_j)^2
    for i in range(N_CIRCLES):
        for j in range(i + 1, N_CIRCLES):
            d2 = np.sum((c[i] - c[j]) ** 2)
            rs = r[i] + r[j]
            cons.append(d2 - rs * rs)
            
    return np.array(cons)

def initial_guess(seed):
    """Generate a feasible initial configuration using a hexagonal lattice."""
    rng = np.random.RandomState(seed)
    pts = []
    s = 0.15
    # Hexagonal/triangular grid covers the square efficiently
    for i in range(15):
        for j in range(15):
            x = i * s + (j % 2) * s / 2
            y = j * s * np.sqrt(3) / 2
            if 0.05 <= x <= 0.95 and 0.05 <= y <= 0.95:
                pts.append([x, y])
                
    if len(pts) < N_CIRCLES:
        pts = rng.rand(N_CIRCLES, 2) * 0.8 + 0.1
    else:
        idx = rng.choice(len(pts), N_CIRCLES, replace=False)
        pts = np.array(pts)[idx]
        
    r = np.ones(N_CIRCLES) * 0.04
    return np.concatenate([pts.flatten(), r])

def run_packing():
    bounds = [(0, 1)] * (2 * N_CIRCLES) + [(0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    best_x = None
    best_sum = -1.0
    
    # Multi-restart to escape local optima
    for s in range(5):
        x0 = initial_guess(s)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                          constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
            
            # Verify feasibility with tolerance
            if np.all(get_constraints(res.x) >= -1e-6):
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_x = res.x
        except Exception:
            continue
            
    if best_x is None:
        best_x = initial_guess(42)
        
    c = best_x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    r = best_x[2 * N_CIRCLES:]
    
    # Strict validation and safe shrinking if numerical errors occur
    while True:
        valid = True
        for i in range(N_CIRCLES):
            if (r[i] > c[i, 0] + 1e-9 or r[i] > c[i, 1] + 1e-9 or 
                r[i] > 1 - c[i, 0] + 1e-9 or r[i] > 1 - c[i, 1] + 1e-9):
                valid = False
                break
        if valid:
            for i in range(N_CIRCLES):
                for j in range(i + 1, N_CIRCLES):
                    d = np.sqrt(np.sum((c[i] - c[j]) ** 2))
                    if d < r[i] + r[j] - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            break
        r *= 0.995 # Gradually shrink until strictly valid
        
    return c, r, np.sum(r)
