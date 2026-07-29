# sol_000030 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000016 (state 3dc87422) state=8a6c7795 sum of radii=2.537203 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(v):
    """Objective: Minimize negative sum of radii"""
    return -np.sum(v[2*N:])

def constraints(v):
    """Constraints: Boundary and non-overlap"""
    c = v[:2*N].reshape(N, 2)
    r = v[2*N:]
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    bc = np.concatenate([c[:,0]-r, 1-c[:,0]-r, c[:,1]-r, 1-c[:,1]-r])
    
    # Non-overlap constraints: dist(i,j) >= r_i + r_j
    diff = c[:, None, :] - c[None, :, :]
    d = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(d, 1.0)
    rs = r[:, None] + r[None, :]
    idx = np.triu_indices(N, k=1)
    oc = d[idx] - rs[idx]
    
    return np.concatenate([bc, oc])

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]"""
    b = []
    for _ in range(N):
        b.append((0.0, 1.0))
        b.append((0.0, 1.0))
        b.append((0.0, 0.5))
    return b

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_c = None
    best_r = None
    
    inits = []
    # 1. Hexagonal pattern
    hc = []
    y = 0.1
    r_idx = 0
    while len(hc) < N:
        x = 0.1 if r_idx % 2 == 0 else 0.2
        while x <= 0.9 and len(hc) < N:
            hc.append([x, y])
            x += 0.2
        y += 0.15
        r_idx += 1
    hc = np.array(hc[:N])
    inits.append(np.concatenate([hc.flatten(), np.full(N, 0.05)]))
    
    # 2. Random configurations
    for _ in range(6):
        rc = np.random.rand(N, 2) * 0.8 + 0.1
        inits.append(np.concatenate([rc.flatten(), np.full(N, 0.05)]))
        
    # 3. Grid pattern
    gc = []
    for i in range(5):
        for j in range(5):
            gc.append([0.1 + i*0.2, 0.1 + j*0.2])
    gc.append([0.5, 0.5])
    gc = np.array(gc[:N])
    inits.append(np.concatenate([gc.flatten(), np.full(N, 0.05)]))
    
    # 4. Corner-focused
    cc = [[0.1,0.1], [0.9,0.1], [0.1,0.9], [0.9,0.9]]
    for i in range(6):
        for j in range(6):
            cc.append([0.2 + i*0.12, 0.2 + j*0.12])
    cc = np.array(cc[:N])
    inits.append(np.concatenate([cc.flatten(), np.full(N, 0.05)]))
    
    # Optimize from each initialization
    for x0 in inits:
        x0 = x0 + np.random.normal(0, 0.001, x0.shape)
        x0 = np.clip(x0, 0.0, 1.0)
        x0[-N:] = np.clip(x0[-N:], 0.0, 0.5)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 1500, 'ftol': 1e-10, 'disp': False})
            if res.success or -res.fun > best_sum:
                v = res.x
                cvals = constraints(v)
                if np.all(cvals >= -1e-7):
                    r = v[2*N:]
                    if np.all(r >= -1e-7):
                        s = np.sum(r)
                        if s > best_sum:
                            best_sum = s
                            best_c = v[:2*N].reshape(N, 2).copy()
                            best_r = r.copy()
        except Exception:
            pass
            
    # Local perturbation search on best solution to escape local minima
    if best_c is not None:
        for _ in range(10):
            x0 = np.concatenate([best_c.flatten(), best_r])
            x0 += np.random.normal(0, 0.005, x0.shape)
            x0 = np.clip(x0, 0.0, 1.0)
            x0[-N:] = np.clip(x0[-N:], 0.0, 0.5)
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False})
                if res.success:
                    v = res.x
                    cvals = constraints(v)
                    if np.all(cvals >= -1e-7):
                        s = -res.fun
                        if s > best_sum:
                            best_sum = s
                            best_c = v[:2*N].reshape(N, 2).copy()
                            best_r = v[2*N:].copy()
            except Exception:
                pass
                
    if best_c is None:
        best_c, best_r = hc, np.full(N, 0.05)
        best_sum = np.sum(best_r)
        
    return best_c, best_r, float(best_sum)
