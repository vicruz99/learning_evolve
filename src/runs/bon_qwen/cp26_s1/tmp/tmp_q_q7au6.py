import numpy as np
from scipy.optimize import minimize

N = 26

def _objective(x):
    return -np.sum(x[2*N:])

def _constraints(x):
    c = x[:2*N].reshape((N, 2))
    r = x[2*N:]
    cons = []
    for i in range(N):
        cons.append(c[i, 0] - r[i])
        cons.append(c[i, 1] - r[i])
        cons.append(1.0 - c[i, 0] - r[i])
        cons.append(1.0 - c[i, 1] - r[i])
    for i in range(N):
        for j in range(i+1, N):
            dx = c[i, 0] - c[j, 0]
            dy = c[i, 1] - c[j, 1]
            cons.append(dx*dx + dy*dy - (r[i] + r[j])**2)
    return np.array(cons)

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': _constraints}
    
    best_sum = -1.0
    best_x = None
    
    for seed in range(20):
        np.random.seed(seed)
        if seed < 12:
            pts = []
            for i in range(6):
                for j in range(5):
                    x = 0.1 + i * 0.15
                    y = 0.1 + j * 0.2 + (0.08 if i % 2 else 0.0)
                    pts.append([x, y])
            np.random.shuffle(pts)
            c0 = np.array(pts[:N])
            r0 = np.full(N, 0.07)
        else:
            c0 = np.random.uniform(0.15, 0.85, (N, 2))
            r0 = np.full(N, 0.05)
            
        x0 = np.zeros(3*N)
        for i in range(N):
            x0[3*i] = c0[i, 0]
            x0[3*i+1] = c0[i, 1]
            x0[3*i+2] = r0[i]
            
        x0[:2*N] += np.random.uniform(-0.02, 0.02, 2*N)
        x0[:N] = np.clip(x0[:N], 0.05, 0.95)
        x0[N:2*N] = np.clip(x0[N:2*N], 0.05, 0.95)
        
        try:
            res = minimize(_objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                val = np.sum(res.x[2*N:])
                if np.all(_constraints(res.x) >= -1e-7):
                    if val > best_sum:
                        best_sum = val
                        best_x = res.x.copy()
        except Exception:
            continue
            
    if best_x is None:
        c = np.random.rand(N, 2) * 0.8 + 0.1
        r = np.full(N, 0.01)
        return c, r, np.sum(r)
        
    c_final = best_x[:2*N].reshape((N, 2))
    r_final = best_x[2*N:]
    return c_final, r_final, best_sum