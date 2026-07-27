# sol_000104 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a3c1a30f) state=09fc2aa3 sum of radii=2.331830 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(v):
    # Maximize sum of radii -> minimize negative sum
    # v structure: [x1, y1, x2, y2, ..., x26, y26, r1, r2, ..., r26]
    return -np.sum(v[52:])

def _constraints(v):
    N = 26
    centers = v[:52].reshape(N, 2)
    radii = v[52:]
    
    # 4 constraints per circle for boundaries + N*(N-1)/2 for pairwise
    # Total: 4*26 + 325 = 429
    cons = np.empty(429)
    k = 0
    
    # Boundary constraints
    for i in range(N):
        x, y = centers[i]
        r = radii[i]
        cons[k] = x - r; k += 1
        cons[k] = 1.0 - x - r; k += 1
        cons[k] = y - r; k += 1
        cons[k] = 1.0 - y - r; k += 1
        
    # Pairwise non-overlap constraints
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx * dx + dy * dy
            sum_r = radii[i] + radii[j]
            cons[k] = dist_sq - sum_r * sum_r
            k += 1
            
    return cons

def run_packing():
    N = 26
    
    # Initialize centers in a grid pattern with slight hexagonal staggering
    centers_init = np.zeros((N, 2))
    idx = 0
    y = 0.1
    shift = 0.0
    while idx < N:
        x = 0.1 + shift
        while x <= 0.9 - shift and idx < N:
            centers_init[idx] = [x, y]
            idx += 1
            x += 0.2
        y += 0.17320508
        shift = 0.1 - shift
        
    # If grid didn't fill all (should for N=26 with this pattern), fill remaining
    while idx < N:
        centers_init[idx] = [0.5, 0.5 + 0.15 * (idx - 26)]
        idx += 1
        
    # Flatten and add initial radii
    v0 = np.zeros(78)
    v0[:52] = centers_init.flatten()
    v0[52:] = 0.09  # Initial feasible radius
    
    # Bounds for optimization
    bounds = [(0.0, 1.0)] * 52 + [(0.0, 0.5)] * 26
    
    cons = {'type': 'ineq', 'fun': _constraints}
    
    # Run SLSQP optimization
    res = minimize(_objective, v0, method='SLSQP', bounds=bounds, 
                   constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
    
    centers = res.x[:52].reshape(N, 2)
    radii = res.x[52:]
    
    # Strict post-processing to guarantee validity
    for i in range(N):
        r = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        for j in range(N):
            if i != j:
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < 1e-12: 
                    d = 1e-12
                r = min(r, d / 2.0)
        radii[i] = r
        
    total_sum = float(np.sum(radii))
    return centers, radii, total_sum
