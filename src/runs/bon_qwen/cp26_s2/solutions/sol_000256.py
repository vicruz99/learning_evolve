# sol_000256 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2b83b1fb) state=545ebd7a sum of radii=2.609952 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

N = 26
EPS = 1e-7

def objective(vars):
    # Variables structure: [x0, y0, ..., x25, y25, r0, r1, ..., r25]
    r = vars[2*N:]
    return -np.sum(r)

def constraint_fun(vars):
    c = vars[:2*N].reshape(N, 2)
    r = vars[2*N:]
    
    # 4 boundary constraints per circle + 1 per pair
    num_constraints = 4*N + N*(N-1)//2
    cons = np.zeros(num_constraints)
    k = 0
    
    # Boundary constraints
    for i in range(N):
        cons[k] = c[i, 0] - r[i] - EPS; k += 1
        cons[k] = 1.0 - c[i, 0] - r[i] - EPS; k += 1
        cons[k] = c[i, 1] - r[i] - EPS; k += 1
        cons[k] = 1.0 - c[i, 1] - r[i] - EPS; k += 1
        
    # Pairwise non-overlap constraints
    for i in range(N):
        diff = c - c[i]
        dist_sq = np.sum(diff**2, axis=1)
        rad_sum = r + r[i]
        for j in range(i + 1, N):
            cons[k] = dist_sq[j] - (rad_sum[j] + EPS)**2
            k += 1
            
    return cons

def run_packing():
    # Initialization with hexagonal packing pattern
    # Centers are placed as if radius was 0.1 to ensure good spatial distribution
    # Optimization starts with r=0.05 to guarantee a strictly feasible initial point
    r_init = 0.05
    centers = np.zeros((N, 2))
    radii = np.full(N, r_init)
    
    r_temp = 0.1
    counts = [5, 4, 5, 4, 5, 3]  # Sums to 26
    idx = 0
    for row in range(len(counts)):
        y = r_temp + row * math.sqrt(3) * r_temp
        count = counts[row]
        for c in range(count):
            x = r_temp + c * 2 * r_temp
            if row % 2 == 1:
                x += r_temp
            if idx < N:
                centers[idx] = [x, y]
                idx += 1
                
    x0 = np.concatenate([centers.flatten(), radii])
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    constraints = [{'type': 'ineq', 'fun': constraint_fun}]
    
    # Optimize using SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                   constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})
    
    final_centers = res.x[:2*N].reshape(N, 2)
    final_radii = res.x[2*N:]
    
    return final_centers, final_radii, np.sum(final_radii)
