# sol_000269 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4c81ffe0) state=37af5ac0 sum of radii=1.663440 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    np.random.seed(42)
    N = 26
    
    # 1. Initialize centers on a perturbed hexagonal grid
    pts = []
    for i in range(6):
        for j in range(5):
            x = 0.08 + i * 0.16 + (j % 2) * 0.08
            y = 0.08 + j * 0.18
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
    centers_init = np.array(pts[:N])
    
    # Add small random jitter to break symmetry
    centers_init += np.random.uniform(-0.005, 0.005, centers_init.shape)
    centers_init = np.clip(centers_init, 0.05, 0.95)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = centers_init[i, 0]
        x0[3*i+1] = centers_init[i, 1]
        x0[3*i+2] = 0.035  # Initial radius
        
    # Bounds for variables
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    
    def objective(vars):
        C = vars[:2*N].reshape(N, 2)
        R = vars[2*N:]
        
        # Objective: maximize sum of radii -> minimize negative sum
        obj = -np.sum(R)
        
        # Vectorized pairwise distances
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        D = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(D, np.inf)
        
        # Overlap penalty
        S = R[:, np.newaxis] + R[np.newaxis, :]
        olap = np.maximum(0, S - D)
        obj += 5000.0 * np.sum(olap**2)
        
        # Boundary penalties
        obj += 5000.0 * np.sum(np.maximum(0, R - C[:, 0])**2)
        obj += 5000.0 * np.sum(np.maximum(0, C[:, 0] + R - 1)**2)
        obj += 5000.0 * np.sum(np.maximum(0, R - C[:, 1])**2)
        obj += 5000.0 * np.sum(np.maximum(0, C[:, 1] + R - 1)**2)
        
        return obj

    # 2. Optimize using L-BFGS-B
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 100000, 'ftol': 1e-16})
                   
    C = res.x[:2*N].reshape(N, 2)
    R = res.x[2*N:]
    
    # 3. Post-processing: strict constraint satisfaction
    C = np.clip(C, 1e-8, 1 - 1e-8)
    
    # Enforce boundary limits on radii
    for i in range(N):
        max_r = min(C[i,0], 1-C[i,0], C[i,1], 1-C[i,1])
        R[i] = min(R[i], max_r - 1e-9)
        
    # Iteratively shrink overlapping circles to guarantee validity
    for _ in range(500):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(C[i] - C[j])
                if d < R[i] + R[j] - 1e-10:
                    shrink = (R[i] + R[j] - d) / 2.0 + 1e-10
                    R[i] -= shrink
                    R[j] -= shrink
                    changed = True
        if not changed:
            break
            
    # Ensure non-negative radii
    R = np.maximum(R, 0.0)
    
    return C, R, float(np.sum(R))
