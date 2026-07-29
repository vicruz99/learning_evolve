# sol_000057 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000030 (state 57c93ce5) state=88881a25 sum of radii=2.623068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
_OVERLAP_IDX = np.tril_indices(N_CIRCLES, -1)

def objective(vars):
    """Objective to minimize: negative sum of radii."""
    return -np.sum(vars[2::3])

def constraints(vars):
    """Vectorized inequality constraints: g(vars) >= 0."""
    n = N_CIRCLES
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dr = r[:, np.newaxis] + r[np.newaxis, :]
    
    dist_sq = dx**2 + dy**2
    r_sum_sq = dr**2
    
    i_idx, j_idx = _OVERLAP_IDX
    c = np.concatenate([c, dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx]])
    return c

def generate_init(seed, method):
    """Generates a strictly feasible initial configuration."""
    np.random.seed(seed)
    n = N_CIRCLES
    centers = np.zeros((n, 2))
    
    if method == 'hex':
        angle = np.random.uniform(0.0, 0.6)
        r_base = 0.1
        temp = []
        y = r_base
        row = 0
        while len(temp) < n + 5:
            x = r_base if row % 2 == 0 else 2 * r_base
            while len(temp) < n + 5:
                if x + r_base > 1.1:
                    break
                temp.append([x, y])
                x += 2 * r_base
            y += np.sqrt(3) * r_base
            row += 1
            if y + r_base > 1.1:
                break
                
        centers = np.array(temp[:n])
        centers -= centers.min(axis=0)
        centers /= centers.max(axis=0)
        centers = centers * 0.8 + 0.1
        
        cx, cy = 0.5, 0.5
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        dx = centers[:, 0] - cx
        dy = centers[:, 1] - cy
        centers[:, 0] = dx * cos_a - dy * sin_a + cx
        centers[:, 1] = dx * sin_a + dy * cos_a + cy
    else:
        centers = np.random.uniform(0.15, 0.85, (n, 2))
        
    centers = np.clip(centers, 0.05, 0.95)
    radii = np.zeros(n)
    
    # Compute safe initial radii to guarantee feasibility
    for i in range(n):
        min_d = min(centers[i, 0], 1.0 - centers[i, 0],
                    centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(centers[i, 0] - centers[j, 0],
                             centers[i, 1] - centers[j, 1])
                if d < min_d:
                    min_d = d
        radii[i] = min_d / 2.0 * 0.85
        
    vars = np.zeros(3 * n)
    vars[0::3] = centers[:, 0]
    vars[1::3] = centers[:, 1]
    vars[2::3] = radii
    return vars

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_vars = None
    
    # Phase 1: Diverse multi-start optimization
    for seed in range(40):
        method = 'hex' if seed % 3 != 0 else 'rand'
        x0 = generate_init(seed, method)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 50000, 'ftol': 1e-12, 'disp': False})
            s = -res.fun
            if not np.isnan(s) and s > best_sum:
                c_vals = constraints(res.x)
                if np.min(c_vals) >= -1e-7:
                    best_sum = s
                    best_vars = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local refinement to escape shallow minima
    if best_vars is not None:
        for _ in range(5):
            x0 = best_vars + np.random.normal(0, 5e-5, 3 * n)
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 30000, 'ftol': 1e-12, 'disp': False})
                s = -res.fun
                if not np.isnan(s) and s > best_sum:
                    c_vals = constraints(res.x)
                    if np.min(c_vals) >= -1e-7:
                        best_sum = s
                        best_vars = res.x.copy()
            except Exception:
                break
                
    # Fallback
    if best_vars is None:
        best_vars = generate_init(0, 'hex')
        
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    centers[:, 0] = best_vars[0::3]
    centers[:, 1] = best_vars[1::3]
    radii[:] = best_vars[2::3]
    
    # Final safety repair to guarantee strict validity against validation function
    valid = True
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        if x - r < -1e-10 or x + r > 1 + 1e-10 or y - r < -1e-10 or y + r > 1 + 1e-10:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-10:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        factor = 0.99
        for _ in range(100):
            radii *= factor
            centers[:, 0] = np.clip(centers[:, 0], radii, 1 - radii)
            centers[:, 1] = np.clip(centers[:, 1], radii, 1 - radii)
            ok = True
            for i in range(n):
                for j in range(i + 1, n):
                    if np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1]) < radii[i] + radii[j] - 1e-12:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                break
                
    return centers, radii, float(np.sum(radii))
