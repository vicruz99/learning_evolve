# sol_000262 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8d1f387b) state=220b51ff sum of radii=2.556021 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Optimizes the placement and radii of 26 circles in a unit square 
    to maximize the sum of radii.
    """
    N = 26
    
    # 1. Initialize with a hexagonal grid pattern
    centers = np.zeros((N, 2))
    idx = 0
    for row in range(7):
        y = 0.08 + row * 0.14
        if y > 0.92: 
            break
        start_x = 0.08 if row % 2 == 0 else 0.15
        for col in range(6):
            x = start_x + col * 0.16
            if x > 0.92: 
                break
            if idx < N:
                centers[idx] = [x, y]
                idx += 1
            else:
                break
        if idx >= N: 
            break
            
    # Fallback fill if fewer than 26 points generated
    while idx < N:
        centers[idx] = [0.5, 0.5]
        idx += 1
        
    radii = np.full(N, 0.095)
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds for variables: x,y in [0,1], r in [0, 0.2]
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.2)] * N
    
    # Objective: Minimize negative sum of radii
    def obj(x):
        return -np.sum(x[2*N:])
        
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    def con_boundary(x):
        c = x[:2*N].reshape((N, 2))
        r = x[2*N:]
        v = np.zeros(4*N)
        v[0::4] = c[:, 0] - r          # x >= r
        v[1::4] = 1.0 - c[:, 0] - r    # x + r <= 1
        v[2::4] = c[:, 1] - r          # y >= r
        v[3::4] = 1.0 - c[:, 1] - r    # y + r <= 1
        return v
        
    # Overlap constraints: distance between centers >= sum of radii
    def con_overlap(x):
        c = x[:2*N].reshape((N, 2))
        r = x[2*N:]
        v = np.zeros(N*(N-1)//2)
        k = 0
        for i in range(N):
            for j in range(i+1, N):
                v[k] = np.sqrt(np.sum((c[i] - c[j])**2)) - r[i] - r[j]
                k += 1
        return v
        
    cons = [
        {'type': 'ineq', 'fun': con_boundary},
        {'type': 'ineq', 'fun': con_overlap}
    ]
    
    # 2. Run SLSQP optimization
    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons,
                   options={'maxiter': 5000, 'ftol': 1e-12})
    
    c_opt = res.x[:2*N].reshape((N, 2))
    r_opt = res.x[2*N:]
    
    # 3. Final validation check and safety shrinkage if needed
    min_gap = 1.0
    for i in range(N):
        min_gap = min(min_gap, 
                      c_opt[i,0] - r_opt[i], 1.0 - c_opt[i,0] - r_opt[i],
                      c_opt[i,1] - r_opt[i], 1.0 - c_opt[i,1] - r_opt[i])
    for i in range(N):
        for j in range(i+1, N):
            d = np.sqrt(np.sum((c_opt[i] - c_opt[j])**2))
            min_gap = min(min_gap, d - r_opt[i] - r_opt[j])
            
    # If slight numerical violation occurs, scale radii down minimally
    if min_gap < -1e-10:
        scale = 1.0 - abs(min_gap) * 3.0
        r_opt = np.maximum(r_opt * scale, 0.0)
        
    return c_opt, r_opt, float(np.sum(r_opt))
