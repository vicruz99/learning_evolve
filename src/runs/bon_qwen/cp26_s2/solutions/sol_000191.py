# sol_000191 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 624944be) state=bda98310 sum of radii=2.470427 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars, n):
    return -np.sum(vars[2::3])

def compute_constraints(vars, n):
    cons = np.zeros(4 * n + n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        cons[idx] = vars[3*i] - vars[3*i+2]
        idx += 1
        cons[idx] = 1.0 - (vars[3*i] + vars[3*i+2])
        idx += 1
        cons[idx] = vars[3*i+1] - vars[3*i+2]
        idx += 1
        cons[idx] = 1.0 - (vars[3*i+1] + vars[3*i+2])
        idx += 1
    for i in range(n):
        for j in range(i + 1, n):
            dx = vars[3*i] - vars[3*j]
            dy = vars[3*i+1] - vars[3*j+1]
            cons[idx] = np.hypot(dx, dy) - (vars[3*i+2] + vars[3*j+2])
            idx += 1
    return cons

def generate_initial(n, seed):
    rng = np.random.RandomState(seed)
    centers = []
    y = 0.1
    while len(centers) < n:
        x = 0.1
        while len(centers) < n:
            centers.append([x, y])
            x += 0.18
        y += 0.16
        x = 0.18 if (y - 0.1) / 0.16 % 2 < 0.5 else 0.1
    centers = np.array(centers[:n])
    centers += rng.uniform(-0.02, 0.02, size=centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    radii = np.full(n, 0.04)
    return centers, radii

def run_single_opt(n, seed):
    c0, r0 = generate_initial(n, seed)
    x0 = np.zeros(3 * n)
    x0[0::3] = c0[:, 0]
    x0[1::3] = c0[:, 1]
    x0[2::3] = r0
    
    bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    
    res = minimize(compute_objective, x0, args=(n,), method='SLSQP',
                   bounds=bounds, constraints=cons,
                   options={'maxiter': 2500, 'ftol': 1e-10})
    
    return res.x, np.sum(res.x[2::3]), res.success

def run_packing():
    n = 26
    best_x = None
    best_sum = 0.0
    
    for seed in [42, 123, 456, 789, 999, 1000, 2000]:
        x, s, _ = run_single_opt(n, seed)
        if s > best_sum:
            best_sum = s
            best_x = x
            
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3].copy()
    
    # Ensure strict feasibility within numerical tolerance
    eps = 1e-6
    centers[:, 0] = np.clip(centers[:, 0], radii + eps, 1.0 - radii - eps)
    centers[:, 1] = np.clip(centers[:, 1], radii + eps, 1.0 - radii - eps)
    
    # Iterative radius shrinkage if overlaps persist due to numerical issues
    for _ in range(20):
        min_gap = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                g = d - radii[i] - radii[j]
                if g < min_gap:
                    min_gap = g
        if min_gap < -eps:
            radii *= 0.99
            centers[:, 0] = np.clip(centers[:, 0], radii + eps, 1.0 - radii - eps)
            centers[:, 1] = np.clip(centers[:, 1], radii + eps, 1.0 - radii - eps)
        else:
            break
            
    return centers, radii, np.sum(radii)
