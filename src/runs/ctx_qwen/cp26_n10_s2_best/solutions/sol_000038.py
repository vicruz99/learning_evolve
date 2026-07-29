# sol_000038 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000028 (state 1c5b6a86) state=a156f039 sum of radii=2.625390 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def _objective(v, n):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(v[2*n:])

def _constraints(v, n, p_i, p_j):
    """Inequality constraints: boundaries and non-overlap."""
    c = v[:2*n].reshape(n, 2)
    r = v[2*n:]
    cons = [
        c[:, 0] - r,
        1.0 - c[:, 0] - r,
        c[:, 1] - r,
        1.0 - c[:, 1] - r,
        np.sum((c[p_i] - c[p_j])**2, axis=1) - (r[p_i] + r[p_j])**2
    ]
    return np.concatenate(cons)

def run_packing():
    n = 26
    # Precompute pair indices for vectorized constraints
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    p_i = np.array([p[0] for p in pairs])
    p_j = np.array([p[1] for p in pairs])
    
    bounds = [(0.0, 1.0)]*(2*n) + [(0.0, 0.5)]*n
    
    best_sum = -1.0
    best_v = None
    
    starts = []
    # 1. Hexagonal lattice configurations with varying densities & jitter
    for seed in range(6):
        np.random.seed(seed)
        centers = []
        y = 0.09
        row = 0
        while len(centers) < n:
            x = 0.09 if row % 2 == 0 else 0.18
            while x <= 0.91 and len(centers) < n:
                centers.append([x, y])
                x += 0.18
            y += 0.09 * np.sqrt(3)
            row += 1
        centers = np.array(centers[:n])
        centers += np.random.uniform(-0.005, 0.005, centers.shape)
        centers = np.clip(centers, 0.02, 0.98)
        starts.append(np.concatenate([centers.flatten(), np.full(n, 0.06)]))
        
    # 2. Random uniform starts
    for seed in range(5):
        np.random.seed(seed)
        centers = np.random.uniform(0.05, 0.95, (n, 2))
        starts.append(np.concatenate([centers.flatten(), np.full(n, 0.05)]))
        
    # 3. Grid-based starts
    for seed in range(3):
        np.random.seed(seed)
        centers = []
        for i in range(6):
            for j in range(5):
                if len(centers) < n:
                    centers.append([0.1 + i*0.15, 0.1 + j*0.17])
        centers = np.array(centers[:n])
        centers += np.random.uniform(-0.01, 0.01, centers.shape)
        centers = np.clip(centers, 0.02, 0.98)
        starts.append(np.concatenate([centers.flatten(), np.full(n, 0.06)]))

    # Primary optimization pass
    for x0 in starts:
        try:
            res = minimize(_objective, x0, args=(n,), method='SLSQP', bounds=bounds,
                           constraints={'type': 'ineq', 'fun': _constraints, 'args': (n, p_i, p_j)},
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_v = res.x.copy()
        except Exception:
            pass
            
    if best_v is None:
        centers = np.random.uniform(0.1, 0.9, (n, 2))
        radii = np.full(n, 0.04)
        return centers, radii, np.sum(radii)

    c_best = best_v[:2*n].reshape(n, 2)
    r_best = best_v[2*n:]
    
    # Local search: perturb best solution and re-optimize to escape local minima
    for _ in range(6):
        c_pert = c_best + np.random.uniform(-0.003, 0.003, c_best.shape)
        c_pert = np.clip(c_pert, 0.02, 0.98)
        x_pert = np.concatenate([c_pert.flatten(), r_best])
        try:
            res_pert = minimize(_objective, x_pert, args=(n,), method='SLSQP', bounds=bounds,
                                constraints={'type': 'ineq', 'fun': _constraints, 'args': (n, p_i, p_j)},
                                options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if -res_pert.fun > best_sum:
                best_sum = -res_pert.fun
                best_v = res_pert.x.copy()
                c_best = best_v[:2*n].reshape(n, 2)
                r_best = best_v[2*n:]
        except Exception:
            pass

    # Post-processing to guarantee strict validity per validator tolerances
    c_final = np.clip(c_best, 0.0, 1.0)
    r_final = r_best.copy()
    
    eps = 1e-9
    for _ in range(20):
        # Enforce boundary constraints
        for i in range(n):
            r_final[i] = min(r_final[i], c_final[i,0]-eps, 1.0-c_final[i,0]-eps, 
                             c_final[i,1]-eps, 1.0-c_final[i,1]-eps)
                             
        # Enforce non-overlap constraints
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dist = np.hypot(c_final[i,0]-c_final[j,0], c_final[i,1]-c_final[j,1])
                if r_final[i] + r_final[j] > dist - eps:
                    shrink = (r_final[i] + r_final[j] - dist) / 2.0 + eps
                    r_final[i] = max(0.0, r_final[i] - shrink)
                    r_final[j] = max(0.0, r_final[j] - shrink)
                    changed = True
        if not changed:
            break
            
    return c_final, r_final, np.sum(r_final)
