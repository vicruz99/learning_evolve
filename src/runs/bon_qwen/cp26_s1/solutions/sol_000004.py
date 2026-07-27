# sol_000004 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6882cd8b) state=c7af941d sum of radii=2.566880 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    def unpack(vars):
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        for i in range(n):
            centers[i, 0] = vars[3*i]
            centers[i, 1] = vars[3*i+1]
            radii[i] = vars[3*i+2]
        return centers, radii

    def objective(vars):
        _, radii = unpack(vars)
        return -np.sum(radii)
    
    def make_constraints():
        cons = []
        for i in range(n):
            # x >= r => x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx] - v[3*idx+2]})
            # x <= 1-r => 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 1.0 - v[3*idx] - v[3*idx+2]})
            # y >= r => y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: v[3*idx+1] - v[3*idx+2]})
            # y <= 1-r => 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i: 1.0 - v[3*idx+1] - v[3*idx+2]})
            
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: math.sqrt((v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2) - (v[3*i+2] + v[3*j+2])
                })
        return cons

    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 0.5))

    best_res = None
    best_sum = -1.0
    
    # 1. Grid Initialization
    np.random.seed(42)
    centers_init = np.random.uniform(0.15, 0.85, size=(n, 2))
    radii_init = np.full(n, 0.02)
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    constraints = make_constraints()
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500})
    if res.success and -res.fun > best_sum:
        best_res = res
        best_sum = -res.fun

    # 2. Dense Grid Initialization
    centers_init = np.zeros((n, 2))
    idx = 0
    for i in range(6):
        for j in range(6):
            if idx < n:
                centers_init[idx, 0] = 0.1 + j * 0.15
                centers_init[idx, 1] = 0.1 + i * 0.15
                idx += 1
    radii_init = np.full(n, 0.02)
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500})
    if res.success and -res.fun > best_sum:
        best_res = res
        best_sum = -res.fun

    # 3. Hexagonal-like Initialization
    centers_init = np.zeros((n, 2))
    idx = 0
    r_init = 0.03
    for i in range(7):
        for j in range(7):
            if idx < n:
                x = 0.15 + j * 0.12 + (0.06 if i % 2 == 1 else 0.0)
                y = 0.15 + i * 0.10
                centers_init[idx, 0] = x
                centers_init[idx, 1] = y
                idx += 1
    radii_init = np.full(n, 0.02)
    x0 = np.zeros(3*n)
    for i in range(n):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = radii_init[i]
        
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500})
    if res.success and -res.fun > best_sum:
        best_res = res
        best_sum = -res.fun

    centers, radii = unpack(best_res.x)
    # Ensure strict validity and non-negative radii
    radii = np.maximum(radii, 0.0)
    centers[:, 0] = np.clip(centers[:, 0], 0.0, 1.0)
    centers[:, 1] = np.clip(centers[:, 1], 0.0, 1.0)
    
    return centers, radii, float(np.sum(radii))
