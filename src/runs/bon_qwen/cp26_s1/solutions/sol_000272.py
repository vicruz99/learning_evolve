# sol_000272 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 085da352) state=e49216ab sum of radii=1.250264 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
TRIU_IDX = np.triu_indices(N, k=1)

def objective(x):
    """Objective: maximize radius r (stored as last element)"""
    return -x[-1]

def constraints(x):
    """Constraints: circles inside [0,1]^2 and non-overlapping"""
    centers = x[:-1].reshape(N, 2)
    r = x[-1]
    
    con = []
    # Boundary constraints
    con.append(centers[:, 0] - r)
    con.append(1.0 - centers[:, 0] - r)
    con.append(centers[:, 1] - r)
    con.append(1.0 - centers[:, 1] - r)
    
    # Non-overlap constraints: distance^2 >= (2r)^2
    diff = centers[:, None, :] - centers[None, :, :]
    d2 = np.sum(diff**2, axis=2)
    con.append(d2[TRIU_IDX] - 4.0*r**2)
    
    return np.concatenate(con)

def run_packing():
    bounds = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_r = 0.0
    best_x = None
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Standard 5x5 grid + center
    s1 = []
    for i in range(5):
        for j in range(5):
            s1.append([0.1 + i*0.2, 0.1 + j*0.2])
    s1.append([0.5, 0.5])
    starts.append(s1)
    
    # 2. Tighter grid (closer to optimal density)
    s2 = []
    for i in range(5):
        for j in range(5):
            s2.append([0.102 + i*0.196, 0.102 + j*0.196])
    s2.append([0.5, 0.5])
    starts.append(s2)
    
    # 3. Hexagonal lattice subset
    s3 = []
    for row in range(6):
        y = 0.085 + row * 0.143
        ncols = 5 if row % 2 == 0 else 4
        offset = 0.115 if row % 2 == 1 else 0.085
        for col in range(ncols):
            x = offset + col * 0.188
            s3.append([x, y])
    starts.append(s3[:N])
    
    # Optimization runs
    for start_c in starts:
        x0 = np.array(start_c).flatten()
        x0 = np.append(x0, 0.099)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 3000, 'ftol': 1e-9})
            if -res.fun > best_r:
                best_r = -res.fun
                best_x = res.x
        except Exception:
            pass
            
    if best_x is None:
        # Safe fallback
        centers = np.zeros((N, 2))
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [0.1 + i*0.2, 0.1 + j*0.2]
                idx += 1
        centers[25] = [0.5, 0.5]
        return centers, np.full(N, 0.1), 2.5
        
    centers = best_x[:-1].reshape(N, 2)
    
    # Refine radius: compute exact maximum feasible r from optimized positions
    min_d = 1.0
    for i in range(N):
        # Distance to boundaries
        min_d = min(min_d, centers[i, 0], 1.0 - centers[i, 0], 
                        centers[i, 1], 1.0 - centers[i, 1])
        # Distance to other circles
        for j in range(i+1, N):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < min_d:
                min_d = d
                
    final_r = min_d / 2.0
    radii = np.full(N, final_r)
    return centers, radii, N * final_r
