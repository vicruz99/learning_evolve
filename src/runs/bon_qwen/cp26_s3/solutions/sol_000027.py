# sol_000027 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dfc1b343) state=03b5cfad sum of radii=2.623068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
PAIR_I, PAIR_J = np.triu_indices(N, k=1)

def obj_func(vars):
    return -np.sum(vars[2*N:])

def con_func(vars):
    c = vars[:2*N].reshape(N, 2)
    r = vars[2*N:]
    
    con = np.empty(4*N + len(PAIR_I))
    con[:N] = c[:, 0] - r
    con[N:2*N] = 1 - c[:, 0] - r
    con[2*N:3*N] = c[:, 1] - r
    con[3*N:4*N] = 1 - c[:, 1] - r
    
    ci = c[PAIR_I]
    cj = c[PAIR_J]
    diffs = ci - cj
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    r_sum = r[PAIR_I] + r[PAIR_J]
    con[4*N:] = dists - r_sum
    return con

def run_packing():
    best_centers = np.zeros((N, 2))
    best_radii = np.zeros(N)
    best_sum = 0.0
    
    bounds = [(0.0, 1.0)]*(2*N) + [(0.0, 0.5)]*N
    cons = {'type': 'ineq', 'fun': con_func}
    
    for seed in range(30):
        rng = np.random.default_rng(seed)
        
        pts = []
        y = 0.1
        while y < 0.9:
            x = 0.1
            shift = (int(y/0.1) % 2) * 0.06
            while x < 0.9:
                pts.append([x+shift, y])
                x += 0.15
            y += 0.1
        while len(pts) < N:
            pts.append([rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)])
        pts = np.array(pts[:N])
        
        centers = np.clip(pts + rng.normal(0, 0.01, pts.shape), 0.05, 0.95)
        radii = np.full(N, 0.07)
        
        x0 = np.concatenate([centers.flatten(), radii])
        
        res = minimize(obj_func, x0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 1500, 'ftol': 1e-9})
                       
        c_opt = res.x[:2*N].reshape(N, 2)
        r_opt = res.x[2*N:]
        
        con_vals = con_func(res.x)
        if np.all(con_vals >= -1e-10):
            s = np.sum(r_opt)
            if s > best_sum:
                best_sum = s
                best_centers = c_opt
                best_radii = r_opt
                
    return best_centers, best_radii, best_sum
