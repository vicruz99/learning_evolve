# sol_000067 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 10bf7585) state=505b5294 sum of radii=2.618545 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Objective function: minimize negative sum of radii"""
    return -np.sum(vars[-N:])

def constraints(vars):
    """Non-overlap and boundary constraints: must be >= 0"""
    r = vars[-N:]
    c = vars[:2*N].reshape(N, 2)
    cons = []
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt(np.sum((c[i] - c[j]) ** 2))
            cons.append(dist - r[i] - r[j])
        cons.append(c[i, 0] - r[i])
        cons.append(1.0 - c[i, 0] - r[i])
        cons.append(c[i, 1] - r[i])
        cons.append(1.0 - c[i, 1] - r[i])
    return np.array(cons)

def get_initial_config(seed=0):
    """Generate a feasible hexagonal-like initial configuration"""
    np.random.seed(seed)
    c = np.zeros((N, 2))
    idx = 0
    for row in range(7):
        ncols = 5 if row % 2 == 0 else 4
        if idx + ncols > N:
            ncols = N - idx
        for col in range(ncols):
            x = 0.1 + col * 0.18 + (row % 2) * 0.09
            y = 0.1 + row * 0.155
            c[idx] = [x, y]
            idx += 1
        if idx == N:
            break
    # Small random perturbation to break symmetry and help optimizer
    c += np.random.uniform(-0.02, 0.02, c.shape)
    c = np.clip(c, 0.05, 0.95)
    r = np.full(N, 0.04)  # Start small to guarantee feasibility
    return c, r

def run_packing():
    best_sum = -1.0
    best_x = None
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-6, 0.5)] * N
    
    # Multiple restarts to avoid local minima
    seeds = [0, 1, 2, 3, 4, 5, 6, 7]
    for s in seeds:
        c_init, r_init = get_initial_config(seed=s)
        x0 = np.concatenate([c_init.ravel(), r_init])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            current_sum = -res.fun
            if current_sum > best_sum:
                best_sum = current_sum
                best_x = res.x
        except Exception:
            continue
            
    # Fallback if all optimizations fail
    if best_x is None:
        c_init, r_init = get_initial_config(seed=0)
        best_x = np.concatenate([c_init.ravel(), r_init])
        
    c_opt = best_x[:2*N].reshape(N, 2)
    r_opt = best_x[-N:]
    
    # Safety scaling: ensure strict validity within numerical tolerance
    min_ratio = 1.0
    for i in range(N):
        val1 = c_opt[i, 0] / r_opt[i]
        val2 = (1.0 - c_opt[i, 0]) / r_opt[i]
        val3 = c_opt[i, 1] / r_opt[i]
        val4 = (1.0 - c_opt[i, 1]) / r_opt[i]
        min_ratio = min(min_ratio, val1, val2, val3, val4)
        
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.sqrt(np.sum((c_opt[i] - c_opt[j]) ** 2))
            sum_r = r_opt[i] + r_opt[j]
            if sum_r > 1e-9:
                min_ratio = min(min_ratio, dist / sum_r)
                
    if min_ratio < 1.0:
        r_opt = r_opt * min_ratio * 0.9999999
        
    return c_opt, r_opt, np.sum(r_opt)
