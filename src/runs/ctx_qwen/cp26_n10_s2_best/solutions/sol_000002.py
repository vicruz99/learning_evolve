# sol_000002 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 77dfa116) state=b1ec0bd1 sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def _objective(vars):
    return -np.sum(vars[2*N:])

def _constraints(vars):
    c = vars[:2*N].reshape(N, 2)
    r = vars[2*N:]
    vals = []
    for i in range(N):
        vals.append(c[i,0] - r[i])
        vals.append(1.0 - c[i,0] - r[i])
        vals.append(c[i,1] - r[i])
        vals.append(1.0 - c[i,1] - r[i])
    for i in range(N):
        for j in range(i+1, N):
            d = np.hypot(c[i,0]-c[j,0], c[i,1]-c[j,1])
            vals.append(d - r[i] - r[j])
    return np.array(vals)

def run_packing():
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    
    for attempt in range(5):
        s = 0.2
        pts = []
        for j in range(8):
            for i in range(8):
                x = i * s + (j % 2) * s / 2
                y = j * s * np.sqrt(3) / 2
                if 0 <= x <= 1 and 0 <= y <= 1:
                    pts.append([x, y])
                if len(pts) >= N: break
            if len(pts) >= N: break
            
        centers = np.array(pts[:N])
        centers += np.random.uniform(-0.02, 0.02, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        radii = np.full(N, 0.08)
        
        x0 = np.concatenate([centers.ravel(), radii])
        bounds = [(0, 1)] * (2*N) + [(0, 0.5)] * N
        
        cons_dict = {'type': 'ineq', 'fun': _constraints}
        
        res = minimize(_objective, x0, method='SLSQP', bounds=bounds, constraints=cons_dict, options={'maxiter': 300, 'ftol': 1e-9})
        
        c_opt = res.x[:2*N].reshape(N, 2)
        r_opt = res.x[2*N:]
        
        # Validation and scaling to ensure strict feasibility
        min_scale = 1.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.hypot(c_opt[i,0]-c_opt[j,0], c_opt[i,1]-c_opt[j,1])
                if r_opt[i] + r_opt[j] > 1e-12:
                    min_scale = min(min_scale, d / (r_opt[i] + r_opt[j]))
            if r_opt[i] > 1e-12:
                min_scale = min(min_scale, c_opt[i,0] / r_opt[i])
                min_scale = min(min_scale, (1.0 - c_opt[i,0]) / r_opt[i])
                min_scale = min(min_scale, c_opt[i,1] / r_opt[i])
                min_scale = min(min_scale, (1.0 - c_opt[i,1]) / r_opt[i])
                
        r_opt *= min_scale
        
        s_sum = np.sum(r_opt)
        if s_sum > best_sum:
            best_sum = s_sum
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
            
    return best_centers, best_radii, best_sum
